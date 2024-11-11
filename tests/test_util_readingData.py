# internal
# external
import pytest
# local
from utils.util_readingData import readingDataACL, filter_byLength


def test_equalMapping():
    titles, abstracts = readingDataACL("../dataAnalysis/data/acl_titles_and_abstracts.txt")
    return len(titles) == len(abstracts)


def test_correctOrder():
    titles, abstracts = readingDataACL("../dataAnalysis/data/acl_titles_and_abstracts.txt")
    # builds on the assumption that the titles are on average shorter than the abstract (save to assume)
    return sum([len(t.split()) for t in titles]) / len(titles) <= sum([len(a.split()) for a in abstracts]) / len(
        abstracts)


abstracts = [
        "This is the first abstract.",
        "This abstract is quite a bit longer than the previous one.",
        "Short abstract.",
        "Another example of an abstract text that is fairly lengthy."
    ]

titles = [
    "Title One",
    "An Extended Title",
    "Short",
    "A Long Title for the Last Abstract"
]

def test_filter_too_short():
    filtered_abstracts, filtered_titles = filter_byLength(abstracts, titles, range_abstracts=(5, 100),
                                                          range_titles=(2, 100))

    assert len(filtered_abstracts) == 3  # Only the first three should remain
    assert len(filtered_titles) == 3


def test_filter_too_long():
    filtered_abstracts, filtered_titles = filter_byLength(abstracts, titles, range_abstracts=(0, 10),
                                                          range_titles=(0, 5))

    assert len(filtered_abstracts) == 2  # Only the first two should remain (up to 10 words)
    assert len(filtered_titles) == 2


def test_no_range():
    filtered_abstracts, filtered_titles = filter_byLength(abstracts, titles)

    assert filtered_abstracts == abstracts
    assert filtered_titles == titles


def test_incomplete_sample():
    filtered_abstracts, filtered_titles = filter_byLength(abstracts, titles, range_abstracts=(0, 3),
                                                          range_titles=(2, 3))

    assert len(filtered_abstracts) == 0  # No samples should remain
    assert len(filtered_titles) == 0


def test_mixed_length_filtering():
    filtered_abstracts, filtered_titles = filter_byLength(abstracts, titles, range_abstracts=(2, 10),
                                                          range_titles=(1, 4))

    assert len(filtered_abstracts) == 2  # Only abstracts within the range should remain
    assert len(filtered_titles) == 2  # Corresponding titles should also remain
    assert "This is the first abstract." in filtered_abstracts
    assert "Short" in filtered_titles


if __name__ == '__main__':


    pytest.main()
