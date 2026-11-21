"""Citation expansion: co-citation counting, tagging, fetching, and selection quota."""

import asyncio

from research_agent.models import Paper
from research_agent.retrieval.expansion import (
    count_cocitations,
    expand_by_citations,
    is_weak_expansion,
    pick_seeds,
)
from research_agent.retrieval.selection import select_balanced_papers


def seed(pid: str, refs: list[str], similarity: float = 0.8) -> Paper:
    return Paper(
        openalex_id=f"https://openalex.org/{pid}",
        title=f"Seed paper {pid} about tool use in language models",
        similarity=similarity,
        referenced_works=refs,
        retrieval_reasons=["relevant"],
    )


# --- counting -----------------------------------------------------------------------
def test_count_cocitations_threshold_order_and_once_per_seed():
    seeds = [
        seed("S1", ["W_REACT", "W_TOOLFORMER", "W_TOOLFORMER"]),  # duplicate ref counts once
        seed("S2", ["W_REACT", "W_TOOLFORMER"]),
        seed("S3", ["W_TOOLFORMER", "W_NICHE"]),
    ]
    assert count_cocitations(seeds, min_citers=2) == [("W_TOOLFORMER", 3), ("W_REACT", 2)]


def test_pick_seeds_prefers_most_similar():
    low, high = seed("A", [], similarity=0.6), seed("B", [], similarity=0.9)
    assert pick_seeds([low, high], 1) == [high]


# --- expansion with a fake OpenAlex client ------------------------------------------
class FakeClient:
    def __init__(self, papers: dict[str, Paper]) -> None:
        self.papers = papers
        self.requested: list[list[str]] = []

    async def get_works_by_ids(self, ids):
        self.requested.append(list(ids))
        return [self.papers[i] for i in ids if i in self.papers]


REACT = Paper(openalex_id="https://openalex.org/W_REACT", title="ReAct: Synergizing Reasoning")
TOOLFORMER = Paper(openalex_id="https://openalex.org/W_TOOLFORMER", title="Toolformer")


def run(client, candidates, **overrides):
    kwargs = {"seed_count": 10, "min_citers": 2, "max_papers": 10} | overrides
    return asyncio.run(expand_by_citations(client, candidates, **kwargs))


def test_fetches_new_landmarks_and_tags_them():
    seeds = [seed("S1", ["W_REACT", "W_TOOLFORMER"]), seed("S2", ["W_REACT", "W_TOOLFORMER"])]
    client = FakeClient({"W_REACT": REACT, "W_TOOLFORMER": TOOLFORMER})

    candidates, new = run(client, seeds)

    assert {p.title for p in new} == {"ReAct: Synergizing Reasoning", "Toolformer"}
    assert all(p.retrieval_reasons == ["cited_by_top"] and p.cocited_by == 2 for p in new)
    assert candidates == seeds  # nothing to tag: no seed is co-cited


def test_existing_candidate_is_tagged_not_refetched():
    # S3 is already a candidate AND cited by S1, S2 -> gets the tag; no request for it.
    s3 = seed("S3", [], similarity=0.7)
    seeds = [seed("S1", ["S3"]), seed("S2", ["S3"]), s3]
    client = FakeClient({})

    candidates, new = run(client, seeds)

    tagged = next(p for p in candidates if p.openalex_id.endswith("S3"))
    assert tagged.retrieval_reasons == ["relevant", "cited_by_top"]
    assert tagged.cocited_by == 2
    assert s3.retrieval_reasons == ["relevant"]  # input not mutated
    assert new == [] and client.requested == []


def test_same_paper_under_another_record_is_skipped():
    seeds = [seed("S1", ["W_DUP"]), seed("S2", ["W_DUP"])]
    duplicate = Paper(openalex_id="https://openalex.org/W_DUP", title=seeds[0].title)  # same title
    _, new = run(FakeClient({"W_DUP": duplicate}), seeds)
    assert new == []


def test_max_papers_limits_the_fetch():
    seeds = [seed("S1", ["W_A", "W_B", "W_C"]), seed("S2", ["W_A", "W_B", "W_C"])]
    client = FakeClient({})
    run(client, seeds, max_papers=2)
    assert client.requested == [["W_A", "W_B"]]


def test_nothing_co_cited_means_no_request():
    client = FakeClient({})
    candidates, new = run(client, [seed("S1", ["W_A"]), seed("S2", ["W_B"])])
    assert new == [] and client.requested == []


# --- selection ---------------------------------------------------------------------
def test_cited_by_top_quota_prefers_on_topic_over_most_co_cited():
    # fix 5 eval: LSTM (cited by 8 seeds, sim 0.66) beat BERT for a transformers question
    backbone = Paper(
        openalex_id="W1", cocited_by=8, similarity=0.664, retrieval_reasons=["cited_by_top"]
    )
    landmark = Paper(
        openalex_id="W2", cocited_by=4, similarity=0.74, retrieval_reasons=["cited_by_top"]
    )
    picked = select_balanced_papers(
        [backbone, landmark], target_count=1, quotas={"cited_by_top": 1}
    )
    assert picked == [landmark]


def test_co_citation_breaks_similarity_ties():
    a = Paper(openalex_id="W1", cocited_by=3, similarity=0.7, retrieval_reasons=["cited_by_top"])
    b = Paper(openalex_id="W2", cocited_by=9, similarity=0.7, retrieval_reasons=["cited_by_top"])
    assert select_balanced_papers([a, b], target_count=1, quotas={"cited_by_top": 1}) == [b]


# --- stricter similarity bar for citation-only papers -------------------------------
def test_is_weak_expansion():
    only_cited = Paper(similarity=0.67, retrieval_reasons=["cited_by_top"])
    also_searched = Paper(similarity=0.67, retrieval_reasons=["highly_cited", "cited_by_top"])
    strong = Paper(similarity=0.72, retrieval_reasons=["cited_by_top"])
    unscored = Paper(similarity=None, retrieval_reasons=["cited_by_top"])  # rerank off

    assert is_weak_expansion(only_cited, 0.68)
    assert not is_weak_expansion(also_searched, 0.68)  # keyword search found it too
    assert not is_weak_expansion(strong, 0.68)
    assert not is_weak_expansion(unscored, 0.68)  # can't judge without a score


def test_references_are_not_part_of_json_output():
    paper = seed("S1", ["W_A"])
    assert "referenced_works" not in paper.model_dump()
    assert paper.referenced_works == ["W_A"]  # still available in Python
