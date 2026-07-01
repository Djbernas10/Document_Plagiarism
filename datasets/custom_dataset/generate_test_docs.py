"""Generate the 20 custom-dataset suspicious documents (10 clean + 10 plagiarised)
as plain UTF-8 .txt files plus matching PAN-style ground-truth .xml files.

Builds each document by concatenating "wrapper" prose (original writing) with
verbatim/edited passages copied from source_documents/. Char offsets for the
ground truth are computed programmatically from the actual concatenation, so
they are guaranteed correct.

Run from project root:
    python datasets/custom_dataset/generate_test_docs.py
"""

from pathlib import Path
from xml.sax.saxutils import escape

BASE = Path(__file__).parent
SRC_DIR = BASE / "source_documents"
SUSP_DIR = BASE / "suspicious_documents"
GT_DIR = BASE / "ground_truth"

SUSP_DIR.mkdir(exist_ok=True)
GT_DIR.mkdir(exist_ok=True)


class DocBuilder:
    """Accumulates text and records (offset, length, source_doc, source_offset, type, obfuscation)."""

    def __init__(self):
        self.text = ""
        self.spans = []  # list of dicts

    def add(self, text: str) -> None:
        self.text += text

    def add_plagiarised(
        self,
        text: str,
        source_doc: str,
        source_offset: int,
        ptype: str,
        obfuscation: str,
        source_length: int | None = None,
    ) -> None:
        start = len(self.text)
        self.text += text
        length = len(text)
        self.spans.append(
            dict(
                this_offset=start,
                this_length=length,
                source_reference=source_doc,
                source_offset=source_offset,
                source_length=source_length if source_length is not None else length,
                type=ptype,
                obfuscation=obfuscation,
            )
        )


def source_text(doc_id: str) -> str:
    return (SRC_DIR / f"{doc_id}.txt").read_text(encoding="utf-8")


def find_offset(haystack: str, needle: str, source_doc: str) -> int:
    idx = haystack.find(needle)
    if idx == -1:
        raise ValueError(f"Passage not found verbatim in {source_doc}:\n{needle[:120]!r}")
    return idx


def write_doc(doc_id: str, builder: DocBuilder) -> None:
    (SUSP_DIR / f"{doc_id}.txt").write_text(builder.text, encoding="utf-8")

    lines = [f'<document reference="{doc_id}.txt">']
    for span in builder.spans:
        lines.append(
            "  <feature\n"
            '    name="plagiarism"\n'
            f'    type="{escape(span["type"])}"\n'
            f'    obfuscation="{escape(span["obfuscation"])}"\n'
            f'    this_offset="{span["this_offset"]}"\n'
            f'    this_length="{span["this_length"]}"\n'
            f'    source_reference="{span["source_reference"]}.txt"\n'
            f'    source_offset="{span["source_offset"]}"\n'
            f'    source_length="{span["source_length"]}"\n'
            "  />"
        )
    lines.append("</document>")
    (GT_DIR / f"{doc_id}.xml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{doc_id}: {len(builder.text)} chars, {len(builder.spans)} plagiarised span(s)")


# ---------------------------------------------------------------------------
# 5 CLEAN documents (00001-00005) — original writing, no plagiarism
# ---------------------------------------------------------------------------

CLEAN_DOCS = {
    "suspicious-document00001": """\
Detecting Machine-Paraphrased Text: An Overview of Current Approaches

Introduction

Automated plagiarism detection has traditionally relied on surface-level text matching:
n-gram overlap, fingerprinting and string alignment algorithms that work well when a
plagiarist copies text with little or no modification. Over the last decade, however, the
rise of paraphrasing tools, machine translation systems and now large language models
has made it considerably easier for a person to disguise copied content while preserving
its meaning. This essay surveys how detection systems have evolved in response to this
challenge, distinguishing between lexical, syntactic and semantic detection strategies, and
discusses why no single strategy is sufficient on its own.

Lexical Approaches

The earliest plagiarism detectors compared documents using shared substrings or shared
sets of n-grams. A suspicious document is split into overlapping windows of a fixed size,
each window is hashed, and matching hashes between a suspicious document and a
reference corpus indicate candidate overlap. This approach is computationally cheap and
extremely effective against verbatim copying, since identical text produces identical
hashes. Its weakness is equally obvious: a single synonym substitution, word reordering,
or punctuation change is often enough to break the hash match entirely. Plagiarists who
are aware of this property can defeat lexical detectors with very little effort, simply by
running a passage through a thesaurus-based rewriting tool.

To partially compensate, many systems use smaller n-gram sizes, character-level n-grams
rather than word-level n-grams, or a sliding window with high overlap, trading precision
for recall. Even with these adjustments, lexical methods remain fundamentally blind to
paraphrasing that substantially changes word choice while preserving the underlying idea.

Syntactic Approaches

Syntactic methods attempt to look past individual word choices by comparing the
grammatical structure of sentences. The intuition is that even when a plagiarist replaces
words with synonyms, the syntactic skeleton of a sentence — its dependency tree or
phrase structure — tends to remain similar, because rewriting a sentence's structure
requires more cognitive effort than swapping out vocabulary. Detection systems built on
this idea parse both documents, extract structural features such as part-of-speech
sequences or dependency relations, and compare these features instead of raw words.

This approach captures more obfuscation than pure lexical matching, but it introduces
new costs. Parsing is computationally expensive at corpus scale, and structural similarity
metrics tend to be noisier than exact string matches, increasing the rate of false positives
when two authors independently use a common sentence pattern to discuss the same
topic. Structural methods are also vulnerable to deliberate sentence reordering, where a
plagiarist keeps individual sentences close to verbatim but changes their order within a
paragraph.

Semantic and Embedding-Based Approaches

The current generation of detection systems increasingly relies on semantic
representations rather than surface text. Dense vector embeddings, produced by
pretrained language models, place sentences with similar meaning close together in a
shared vector space regardless of the specific words used to express that meaning. This
allows detectors to identify paraphrased passages that share almost no vocabulary with
their source, as long as the underlying meaning has not changed.

The main practical difficulty with embedding-based approaches is computational cost at
scale: comparing every chunk of a suspicious document against every chunk of a large
reference corpus using dense vectors is far more expensive than a hash lookup. Most
production systems therefore use embeddings only as a second-stage filter, after a
cheaper method such as TF-IDF or n-gram overlap has already narrowed the candidate
pool to a manageable number of source documents. Another concern is that embedding
similarity is a continuous score rather than a binary match, which means systems must
choose a similarity threshold, and that threshold inevitably trades off false positives
against false negatives depending on how aggressively it is set.

A further complication specific to embeddings is that semantic similarity is not the same
as plagiarism. Two independently written passages about the same well-known topic, for
instance a textbook definition of an algorithm, will naturally have high semantic similarity
without any copying having occurred. Distinguishing genuine reuse from coincidental topic
overlap therefore usually requires a secondary verification step, often involving a more
expensive model or even a human reviewer, applied only to the candidates that survive the
initial semantic filter.

Combining Multiple Signals

In practice, no detection pipeline relies on a single signal. Most modern systems combine
several retrieval methods — lexical, structural and semantic — and treat each as a branch
that proposes candidate source documents independently. The candidates from each
branch are then merged, scored, and passed to a final classification stage that decides
whether a given passage actually constitutes plagiarism or merely shares vocabulary or
topic with its candidate source. This multi-branch design reflects a basic trade-off: cheap
methods are fast but blind to certain obfuscation types, while expensive methods catch
more obfuscation but cannot be run exhaustively over a large corpus. Layering them lets a
system get the recall benefits of expensive methods without paying their full computational
cost on every comparison.

The Role of Large Language Models in Detection

A more recent development complicates this picture further. Large language models are
not just a source of obfuscation; they are increasingly used as a detection component in
their own right. A language model can be prompted to compare a suspicious passage
against a candidate source and produce a judgment on whether the passage constitutes
plagiarism, reasoning in natural language about the similarity rather than relying on a
fixed numeric score. This approach can catch obfuscation that earlier methods miss,
because the model brings broad world knowledge and flexible reasoning to the comparison
rather than a narrow similarity metric. It introduces new costs of its own, however: such
models are comparatively slow and expensive to run at the scale of a full corpus
comparison, their judgments are not always consistent across repeated queries on the
same input, and their internal reasoning is difficult to audit compared to a transparent
numeric similarity score. As a result, most pipelines that incorporate a language-model
verification step use it as a final, expensive stage applied only to a small number of
already-shortlisted candidates, mirroring the broader pattern of cheap-filter-then-
expensive-verifier design seen throughout this field.

Corpus-Level Retrieval versus Pairwise Comparison

A separate but related challenge concerns scale: most of this essay has discussed how to
decide whether two specific passages constitute plagiarism, but a deployed system also
needs to decide which of potentially millions of candidate source documents are even
worth comparing against a given suspicious passage in the first place. This retrieval step
typically uses an inverted index or an approximate nearest-neighbour search over
document or chunk-level representations, which trades exhaustive accuracy for the ability
to narrow a huge corpus down to a shortlist of a few dozen or a few hundred candidates in
a fraction of a second. The quality of this initial retrieval step places a hard ceiling on
overall system recall: no matter how sophisticated the downstream comparison method is,
a source document that never makes it into the shortlist can never be correctly identified
as the origin of a plagiarised passage.

Granularity of Comparison

Detection systems must also choose what unit of text to compare: whole documents,
paragraphs, fixed-size sliding windows, or individual sentences. Coarser units such as
whole documents are cheap to compare but blur together plagiarised and original content
within the same document, diluting the similarity signal for a short copied passage
embedded in an otherwise original document. Finer units such as individual sentences
preserve more precise localisation of exactly where the overlap occurs, but multiply the
number of pairwise comparisons that need to be made and increase sensitivity to noise
from sentences that happen to be short or generic. Most practical systems settle on an
intermediate granularity, such as overlapping windows of one or two hundred words, as a
compromise between these two failure modes.

Cross-Lingual Plagiarism Detection

A further complication arises when a plagiarist translates a source passage into a different
language before incorporating it into a new document, a practice often called cross-lingual
plagiarism. Lexical and syntactic methods are essentially powerless against this kind of
obfuscation, since the surface form of the text changes completely even when the
underlying content is reproduced exactly. Detecting cross-lingual reuse generally requires
either translating the suspicious document into the language of the reference corpus
before comparison, or using multilingual embedding models that are explicitly trained so
that semantically equivalent sentences in different languages end up close together in a
shared vector space regardless of which language each one happens to be written in.
Machine translation as a preprocessing step introduces its own source of error, since
translation quality varies considerably across language pairs and domains, and a poor
translation can distort meaning enough to push a genuinely plagiarised passage below a
similarity threshold that would otherwise have flagged it. Multilingual embeddings avoid
the translation step entirely but inherit whatever blind spots the underlying multilingual
model has, which tend to be worse for low-resource language pairs that were underrepresented
during pretraining. Because of these difficulties, cross-lingual plagiarism remains one of
the least reliably detected categories of academic misconduct, and most published
benchmarks in this area report substantially lower recall than equivalent monolingual
benchmarks using otherwise comparable methods.

Detecting Plagiarism in Source Code

Source code plagiarism presents a related but distinct set of challenges that are worth
considering alongside natural-language detection, since many of the same institutions that
worry about essay plagiarism also need to check programming assignments. Code can be
trivially obfuscated through renaming variables, reordering independent statements,
reformatting whitespace, or restructuring control flow into an equivalent but textually
different form, none of which changes program behaviour. Effective code plagiarism
detectors therefore typically normalise the input first, converting source code into an
abstract syntax tree or a token stream stripped of identifier names and formatting, before
applying a similarity measure to this normalised representation. This preprocessing step
plays a role analogous to the lexical and structural methods discussed earlier for natural
language, but the transformation is more aggressive because the space of meaning-preserving
edits in a programming language is larger and more mechanically systematic than in
natural prose, where synonym substitution does not have an exact formal equivalent to
the kind of structural transformation a compiler can apply to code.

Dataset Construction for Detection Research

Progress in this field depends heavily on the availability of benchmark datasets with
reliable ground truth, yet constructing such datasets is itself a non-trivial methodological
challenge. A benchmark built entirely from artificially inserted plagiarism, where
passages are programmatically copied and obfuscated according to a fixed rule, risks
teaching detectors to recognise the signature of the obfuscation procedure itself rather
than plagiarism as it actually occurs among real students or authors. A benchmark built
from real cases of detected misconduct avoids this artificiality but is far more expensive to
construct, since each case requires verified ground truth about which passage actually
originated from which source, information that is rarely available outside a small number
of cases that have already been investigated and confirmed by a human reviewer. Most
published benchmarks therefore combine both approaches, mixing a smaller core of
verified real cases with a larger volume of synthetically generated examples designed to
approximate the variety of obfuscation strategies observed in the verified subset, accepting
the resulting trade-off between authenticity and scale.

Adversarial Dynamics Between Plagiarists and Detectors

It is worth stepping back to note that detection and obfuscation exist in a genuinely
adversarial relationship, where progress on one side tends to provoke a corresponding
adaptation on the other rather than settling into a stable equilibrium. A plagiarist who
becomes aware that a particular detection method is in use has a direct incentive to adopt
whichever obfuscation strategy defeats that specific method, and this dynamic plays out
repeatedly as new detection techniques are published and subsequently studied by the
population they are meant to catch. This adversarial framing has practical implications for
how detection research should be evaluated: a method that performs well against a fixed,
publicly known obfuscation strategy provides much weaker evidence of real-world
usefulness than a method evaluated against an adaptive adversary who is permitted to
adjust their strategy in response to the detector's known weaknesses, since only the latter
setting resembles the conditions under which the method will actually be deployed.

Human-in-the-Loop Verification

Even the most sophisticated automated pipeline rarely makes a final determination of
plagiarism without some form of human review, both because institutional policy usually
requires it and because automated similarity scores capture overlap rather than intent,
and intent is precisely the distinction a misconduct finding typically depends on. A human
reviewer examining a flagged passage can recognise context that no current automated
signal captures well, such as whether a passage is a properly attributed quotation, a
standard formulaic phrase common to an entire subfield, or a genuine instance of
unattributed copying. The practical design question this raises is how to present flagged
candidates to a reviewer in a way that makes efficient use of their limited time: showing
too little context forces the reviewer to do additional investigation themselves, while
showing too much context for every flagged case slows the review process down across
an entire institution's caseload. Most production interfaces settle on showing the
suspicious passage side by side with its candidate source passage, highlighting the specific
overlapping spans, which gives a reviewer enough information to make a fast judgment in
the large majority of clear-cut cases while still allowing them to pull up the full documents
for the harder, more ambiguous ones.

Conclusion

The evolution of plagiarism detection mirrors the evolution of plagiarism itself. As
obfuscation techniques have moved from simple copy-paste to sentence reordering to
fully automated paraphrasing, detection systems have had to move from string matching
to structural analysis to semantic representations, and now increasingly to language-model-
based verification stages layered on top of cheaper retrieval methods. None of these
approaches is complete on its own, and the most effective systems available today are
those that combine multiple complementary signals — operating at a carefully chosen
granularity and balancing retrieval recall against comparison cost — rather than relying on
any single retrieval strategy.
""",
    "suspicious-document00002": """\
Why Citation Practices Differ Across Academic Disciplines

Introduction

Anyone who has read papers from more than one academic field quickly notices that
citation conventions vary enormously. A computer science paper might cite a result with
a short inline reference and a numbered bibliography entry, while a law review article
embeds extensive footnotes that take up half the page. This essay explores some of the
reasons behind these differences, focusing on the relationship between a field's research
methodology, its publication norms, and the citation style that has evolved to support
them.

Citation Density and the Pace of a Field

Fields that move quickly, where results from a few years ago are often considered
outdated, tend to favour citation styles that are compact and forward-looking. Computer
science and physics are good examples: a typical paper might cite dozens of recent
sources but rarely refer back more than five or six years except for genuinely foundational
work. The compact numbered-reference style suits this pattern because it allows authors
to cite many sources without cluttering the body text, and because the underlying claim is
usually "here is the most recent state of the art", which does not require lengthy
discussion of each cited source's argument.

By contrast, fields built on long traditions of accumulated scholarship, such as history or
philosophy, often need to engage at length with how a particular source argued its case,
not just what it concluded. A historian citing a primary source typically needs to specify
exactly which archive, edition or page the claim comes from, and may need to discuss
why that source should be trusted over a conflicting account. This kind of argument does
not fit neatly into a short inline citation, which is part of why footnote-heavy styles
persist in these disciplines even though they are visually denser on the page.

Replicability and the Function of a Citation

In experimental sciences, a citation often serves a very practical function: it points to the
method, dataset or prior result that the current experiment builds on, and ideally allows
another researcher to reproduce or extend the work. This practical orientation favours
citation systems that are easy to parse programmatically, since increasingly these
citations are also consumed by automated tools that build citation graphs, recommend
related papers, or compute bibliometric statistics such as impact factors. A standardised,
machine-readable numbered or author-year format supports all of these downstream
uses far better than a discursive footnote would.

In the humanities, the function of a citation is often closer to attribution and intellectual
honesty than to enabling replication, since the "experiment" being reported is frequently
an interpretive argument rather than a procedure that can be repeated by following a
recipe. This does not make humanities citation less rigorous, but it does mean the citation
needs to carry more context about exactly what is being claimed and on what authority,
which naturally produces longer, more discursive references.

Institutional and Historical Path Dependence

Some of the variation is simply historical inertia. Once a discipline's major journals settle
on a citation style, switching becomes costly: existing typesetting tools, reference
management software defaults, and decades of published norms all reinforce the
incumbent style. This is part of why citation style differences sometimes persist even after
the original methodological reasons for them have weakened. A sub-field that has become
more quantitative over time may still use a citation style inherited from its more
discursive origins, simply because changing the convention would create friction with the
existing body of literature in that journal or society.

Implications for Cross-Disciplinary Work

Researchers working at the boundary between two disciplines, for example digital
humanities or computational social science, often have to navigate two incompatible
citation cultures within the same project. This is more than a cosmetic inconvenience: it
can affect how reviewers from each side evaluate the rigour of a paper, since a reviewer
trained in one tradition may read a sparse citation list as a sign of insufficient engagement
with prior literature, while a reviewer from a citation-dense tradition might read the same
list as appropriately focused. Recognising that citation density is a convention rather than
a universal proxy for rigour is an important, if underappreciated, skill for interdisciplinary
researchers.

Citation Management Tools and Standardisation Pressure

The widespread adoption of reference management software has had a homogenising
effect on citation practice within fields, even as it has done little to close the gap between
fields. Once a journal's preferred style is encoded as a downloadable template for popular
reference managers, authors submitting to that journal face very little friction in matching
its conventions exactly, which has likely reduced the amount of small, idiosyncratic
variation that existed before such tools were common. At the same time, these tools have
made it easier for authors to maintain multiple citation styles for the same body of
references depending on where they intend to submit a given manuscript, which removes
one of the practical incentives a field might otherwise have had to converge toward a
single dominant style shared with neighbouring fields.

Citation Style and the Peer Review Process

Reviewers form judgments about a manuscript's grasp of prior literature partly through how
it cites that literature, and the conventions of a field shape what counts as adequate
engagement in this sense. A field accustomed to short, numerous citations may read a
manuscript that cites only a handful of sources as poorly situated within existing work,
even if those few sources are exactly the right ones, simply because the citation count
itself has become an informal proxy for thoroughness. A field accustomed to fewer, more
deeply discussed citations may read the opposite manuscript, with many brief citations and
little discussion of any single one, as superficial. This means citation style is not a purely
cosmetic matter even within a single field: it actively shapes what reviewers expect to see
and how they evaluate a manuscript's scholarly grounding.

Citation Style as a Marker of Field Identity

Beyond their practical functions, citation conventions sometimes take on a symbolic role,
signalling membership in a particular scholarly community. A paper that uses a citation
style associated with a neighbouring but distinct field, even if the content itself would fit
either field's subject matter, can read as written by an outsider, which may subtly affect
how it is received by reviewers and readers from the "home" field of that style. This
symbolic function helps explain why citation style sometimes persists even when a
sub-field's actual research practices have converged with those of a different discipline:
changing citation style is, in effect, partly a statement about which scholarly community
a piece of work belongs to, not merely a formatting choice.

Citation Practices and the Rise of Preprints

The growing acceptance of preprint servers across many disciplines has introduced a new
wrinkle into citation practice that did not exist when journal publication was the only
legitimate way to disseminate a result. Citing a preprint commits a reader to a version of
a paper that may later change substantially during peer review, or may never be formally
published at all, which raises a question that citation norms inherited from an earlier era
were never designed to answer: should a citation to a preprint be treated as provisional,
and should the published version, once it appears, retroactively replace it in later citing
papers' reference lists. Fields that adopted preprints early, such as physics with its
decades-old culture of posting work before formal review, have developed informal norms
for handling this ambiguity, while fields that have only recently begun to embrace
preprints are still negotiating analogous conventions, often borrowing practices wholesale
from the fields that got there first rather than designing norms specific to their own
publication rhythms.

Self-Citation and Its Interpretation

Self-citation, where an author cites their own prior work, is interpreted very differently
depending on disciplinary norms and on the apparent motivation behind it. In genuinely
cumulative research programmes, where each paper builds incrementally on a research
group's own prior results, a high rate of self-citation can be entirely appropriate and even
expected, since the most relevant prior work genuinely is the author's own. In other
contexts, a conspicuously high rate of self-citation invites suspicion that an author is
attempting to inflate their citation count artificially, particularly in fields where citation
counts feed directly into hiring, promotion or funding decisions. Distinguishing these two
scenarios from the citation record alone is difficult, which is part of why some citation
metrics now explicitly separate self-citations from citations by others when reporting an
author's impact, even though this separation is itself a fairly recent methodological
development that not all bibliometric databases implement consistently.

Citing Negative or Null Results

A field's willingness to cite negative or null results, rather than only positive findings,
says something about how mature its citation culture is with respect to publication bias.
Fields where a negative result is difficult to publish in the first place will naturally show
fewer citations to negative findings, not because such citations are discouraged but
because there are fewer negative findings available to cite. Where venues exist specifically
for null or replication results, citation practice can begin to treat a well-conducted null
result as having genuine evidentiary value worth citing in its own right, rather than
treating the citation record as an implicit register of only what worked, which produces a
systematically more complete picture of a research question's actual evidentiary state for
any later reader who relies on the citation graph to orient themselves.

Citation Practices in Multi-Author, Multi-Disciplinary Teams

Large collaborative projects spanning several disciplines, increasingly common in areas
such as climate science or computational biology, force an explicit negotiation of citation
practice that single-discipline papers rarely require. A genomics specialist contributing a
methods section to a paper otherwise authored by ecologists must decide whether to
follow the citation conventions of the lead discipline, their own home discipline, or some
hybrid compromise, and different co-authors on the same paper may have strong but
conflicting intuitions about which choice signals appropriate rigour. In practice, these
disputes are often resolved by deferring to the convention of whichever discipline the
submission's target journal belongs to, which can leave contributing specialists feeling that
their own discipline's citation norms, and the kind of scholarly engagement those norms
are meant to signal, are being only partially represented in the final published version of
a genuinely collaborative piece of research.

Translations and the Citation of Non-English Scholarship

A frequently overlooked dimension of citation practice concerns scholarship originally
published in languages other than the dominant language of a given field's leading
journals. Citation norms rarely specify how to handle a source that exists only in
translation, or that has never been translated at all, and the practical effect is that
non-English-language scholarship is cited at markedly lower rates even when it is directly
relevant to the topic at hand, simply because most authors and reviewers in a given field
cannot easily evaluate it without translation. This asymmetry compounds over time:
because non-English scholarship accumulates fewer citations, it appears less prominently
in citation-based discovery tools and literature searches, making it still less likely to be
found and cited by the next generation of researchers working in the dominant language,
a feedback loop that has only recently become a subject of explicit methodological
concern within bibliometrics itself.

Citation Practices and Peer Review Anonymity

Double-blind review policies, which ask authors to anonymise their submissions, interact
with citation practice in a way that varies across fields depending on how citation-dense
the field's writing already is. Fields with sparse, selective citation can anonymise more
easily, since omitting or rephrasing a self-citation removes relatively little identifying
information. Fields with dense citation practices face a harder problem: a sufficiently
expert reviewer can often infer an author's identity simply from which prior works are
cited heavily and which are conspicuously omitted, since active researchers in a narrow
subfield tend to be recognisable by their characteristic citation pattern even when explicit
self-citations are removed from the submitted manuscript. This has led some venues in
citation-dense fields to experiment with stricter anonymisation requirements, including
asking authors to cite their own closely related prior work in the third person, though
enforcement of such norms remains inconsistent across reviewers and venues.

Conclusion

Citation practices are not arbitrary stylistic choices; they reflect deeper assumptions about
how a field generates and verifies knowledge, how quickly that knowledge becomes
outdated, and who or what is expected to read and process the citation. They are further
reinforced by the standardisation pressure of citation management tools, by the way
reviewers use citation density as an informal proxy for scholarly thoroughness, and by the
symbolic role citation style plays in signalling membership in a scholarly community.
Understanding these differences helps explain why no single "best" citation style exists
across academic writing as a whole.
""",
    "suspicious-document00003": """\
Reproducibility Challenges in Empirical Software Engineering Research

Introduction

Empirical software engineering papers frequently report experiments comparing tools,
algorithms or development practices, yet a recurring criticism in the field is that many of
these experiments are difficult or impossible for other researchers to reproduce. This essay
examines several structural reasons why reproducibility is harder in software engineering
than it might first appear, and considers what changes could realistically improve the
situation given the constraints researchers actually work under.

The Moving-Target Problem

Unlike many natural sciences, where the object of study is relatively stable across decades,
software engineering studies frequently depend on tools, libraries and platforms that
change rapidly. A study that benchmarks a static analysis tool against a particular version
of a compiler, or that measures developer productivity using a particular IDE plugin, risks
becoming unreproducible within a few years simply because the underlying software
ecosystem has moved on. Reproducing the original experiment exactly would require
reconstructing a snapshot of the entire toolchain, which is rarely documented in sufficient
detail in the original paper.

Data Availability and Proprietary Constraints

Many empirical studies use data drawn from industrial partners: commit histories, bug
reports, code review logs or developer surveys from real companies. Such data is often
subject to confidentiality agreements that prevent its release, even in anonymised form,
which means other researchers cannot rerun the analysis on the same dataset even if the
analysis code itself is shared. This is a genuine tension: the studies with the most realistic,
ecologically valid data are often exactly the ones that cannot be fully reproduced by
outsiders, while fully open datasets sometimes come from less representative contexts
such as student projects or small open-source repositories.

Statistical and Methodological Variation

Even when data and code are both available, subtle methodological choices can produce
different results. Decisions about how to handle outliers, which significance threshold to
use, how to define a "bug-fixing commit" for the purpose of a study, or how to split data
into training and test sets can all shift reported effect sizes meaningfully. Because many of
these choices are made informally during analysis rather than specified in advance, two
careful researchers analysing the same dataset can sometimes reach different conclusions
without either having made a clear error, simply because they made different reasonable
judgment calls along the way.

Incentive Structures in Publication

Reproducibility work itself is rarely rewarded as highly as novel contributions in hiring,
promotion or grant evaluation, which creates a disincentive for researchers to spend time
attempting to replicate prior studies rather than producing new ones. Conferences and
journals have begun introducing artifact evaluation tracks that award badges for
reproducible code and data, which is a meaningful step, but the incentive gap between
producing a new result and verifying an old one remains large in most subfields.

What Realistic Improvement Looks Like

Given these constraints, full bit-for-bit reproducibility is probably not achievable for most
software engineering experiments, and arguably should not be the primary goal. A more
realistic standard is what is sometimes called "reproducibility in spirit": releasing analysis
code, documenting tool versions precisely enough that a close approximation of the
original environment can be rebuilt, and being explicit about which methodological
choices were arbitrary versus principled. Even partial transparency of this kind allows
other researchers to identify which parts of a result are robust to small variations in setup
and which parts are fragile, which is often more useful than an exact replication would be
anyway.

Containerisation as a Partial Solution

Container technologies have offered a meaningful, if incomplete, response to the
moving-target problem. By packaging an experiment's full software environment, down to
specific library versions and operating system configuration, into a portable image, a
research team can substantially extend how long an experiment remains technically
reproducible after publication. Containers do not solve every problem, however: they
typically capture software dependencies but not external services such as third-party APIs
that may change behaviour or be discontinued, and they do nothing about the data
availability or methodological transparency issues discussed earlier. A fully containerised
artifact can therefore still fail to reproduce the original result if it depended on now-
inaccessible external infrastructure, even though every line of code and every dependency
version inside the container itself is preserved exactly.

Negative Results and Publication Bias

A related but distinct obstacle to reproducibility is that failed replication attempts are
themselves difficult to publish. A researcher who attempts to reproduce a prior result and
fails has, in some sense, produced valuable information about the robustness of that
result, but venues for publishing such negative findings are scarce compared to venues for
novel positive results. This creates a one-sided record in the literature: published papers
overwhelmingly describe successful experiments, while failed replication attempts mostly
remain unpublished or are shared informally, if at all. Over time this skews the apparent
reliability of a body of work, since the published literature systematically underrepresents
cases where a result did not hold up under independent scrutiny.

Toward Discipline-Specific Reproducibility Standards

Some subfields of software engineering have begun developing reproducibility checklists
tailored to their specific methodological norms, rather than importing generic
reproducibility standards wholesale from other sciences. A checklist appropriate for a
controlled human-subjects study on developer productivity looks quite different from one
appropriate for a purely computational benchmark comparing two static analysis tools,
since the former needs to address participant recruitment and ethical approval while the
latter needs to address dataset versioning and hardware specification. This move toward
discipline-specific standards reflects a growing recognition that a single generic
reproducibility checklist cannot adequately serve the full methodological diversity found
within software engineering research as a whole.

The Role of Randomness and Non-Determinism

A subtler source of irreproducibility specific to software engineering experiments is the
pervasive presence of non-determinism in the systems under study. Compiler
optimisations, thread scheduling, just-in-time compilation, and even the order in which a
hash table happens to iterate over its keys can all introduce run-to-run variation in
measured performance that has nothing to do with the experimental treatment being
studied. A researcher who runs a benchmark once and reports a single number risks
mistaking ordinary measurement noise for a genuine effect, particularly when the
treatment effect being studied is itself small relative to this background variation. Best
practice in performance-oriented software engineering research calls for repeated runs
with reported variance or confidence intervals, but many published studies still report a
single run per configuration, either because of constrained compute budgets or because
the authors did not anticipate how much noise their particular measurement setup would
introduce.

Tooling Drift in Continuous Integration Pipelines

Many recent empirical studies mine data directly from continuous integration logs,
treating build and test outcomes as a proxy for code quality or developer behaviour. This
introduces a reproducibility hazard distinct from the ones already discussed: the
continuous integration provider's own infrastructure, scheduling policies and even default
timeout values change over time in ways that are rarely versioned or documented
anywhere accessible to outside researchers. A study that measures, for instance, the rate
of flaky test failures across a sample of open-source repositories during a particular month
may be capturing properties of the CI provider's infrastructure at that moment as much as
properties of the test suites themselves, and re-running the same mining script a year
later against the same repositories can produce systematically different results purely
because the underlying execution environment has silently changed underneath the study.

Replication Packages and Their Practical Limitations

The increasing expectation that authors release a replication package alongside a paper
has improved transparency, but the packages themselves vary enormously in how usable
they actually are for an independent researcher attempting to build on them. A package
that consists of a single script with no documentation of expected inputs, dependency
versions or expected runtime forces the would-be replicator to reverse-engineer the
original experimental setup before being able to run anything at all, which in practice
deters most attempts before they begin. Conversely, even a well-documented package can
become unusable within a few years if it depends on a specific operating system version,
a deprecated package index, or institutional infrastructure such as a particular cluster
scheduler that is not available outside the original lab. This means the mere existence of a
replication package is a weak signal of long-term reproducibility on its own; what matters
more is whether the package was designed with portability and longevity in mind from
the outset, which is a property current artifact evaluation processes do not always assess
directly.

Human Subjects Research and the Limits of Exact Replication

Studies involving human participants, such as controlled experiments comparing
programming languages or development practices on novice versus expert developers,
face a reproducibility ceiling that has nothing to do with software tooling at all. The
participant pool itself cannot be replicated: a study run on computer science undergraduates
at one institution in one academic term reflects a particular population with particular
prior coursework and particular levels of motivation tied to whatever course credit or
payment was offered, and even an exact procedural replication run on a different
population may produce different results simply because the populations differ in ways
the original study could not fully characterise or control for. This is not a flaw unique to
software engineering research, but it interacts with the field's already weak reproducibility
norms in a compounding way: a result that fails to replicate could be failing because of
genuine population differences, because of undocumented procedural variation, or
because of tooling drift, and disentangling these three explanations after the fact is rarely
possible from the published record alone.

Meta-Research on Reproducibility Itself

A small but growing body of meta-research treats reproducibility in software engineering
as an empirical question in its own right, systematically sampling published papers from
top venues and attempting to assess what fraction successfully share usable code, usable
data, and sufficient methodological detail to support an independent reanalysis. These
meta-studies consistently find substantial gaps between stated community norms and
actual practice, with a meaningful fraction of papers claiming to provide a replication
package that, upon inspection, is incomplete, undocumented, or no longer executable. This
meta-research is valuable precisely because it provides an evidence base for reproducibility
advocacy that goes beyond individual researchers' anecdotal frustration with failed
replication attempts, giving venues and funding bodies a more rigorous basis for deciding
which policy interventions, such as mandatory artifact evaluation, are actually likely to
move the needle on community-wide practice.

The Role of Open-Source Infrastructure Projects

A distinct but related strand of work examines whether the open-source repositories
themselves, frequently used as the raw material for empirical software engineering
studies, remain stable enough over time to support reanalysis independent of any
particular paper's replication package. A repository's commit history is generally
immutable once published, but the metadata surrounding it, such as issue tracker labels,
continuous integration configuration, or even which commits are considered part of the
main development branch following a history rewrite, can change in ways that alter what
a later researcher extracts when re-running the same mining script against the
ostensibly same repository. Some research groups have responded by archiving frozen
snapshots of the repositories underlying a given study at the time of publication, which
sidesteps this problem for that specific study but does not address the more general
difficulty of comparing newer studies against older ones when both draw on the same
continuously evolving project but at different, undocumented points in its history.

Education and Training as a Slower but Durable Lever

Beyond tooling and policy interventions, a quieter long-term lever for improving
reproducibility is simply how graduate students are trained to conduct empirical research
in the first place. A doctoral student who is taught from the outset to version analysis
scripts, document environment dependencies as a routine part of running an experiment
rather than as an afterthought reserved for the final paper submission, and treat a
negative or inconclusive result as worth recording carefully rather than discarding, carries
those habits into every subsequent project for the remainder of their career. This kind of
cultural change is necessarily slow, since it depends on training pipelines and mentorship
relationships that turn over only as quickly as new cohorts of students enter the field, but
it is also comparatively durable once established, in the sense that it does not depend on
any single venue's policy remaining in place or any single tool continuing to be
maintained.

Conclusion

Reproducibility in software engineering research is constrained by a moving technical
landscape, proprietary data, informal methodological choices and weak incentives, not by
a simple lack of effort on the part of individual researchers. Containerisation, more
deliberate venues for negative results, and discipline-specific reproducibility standards all
offer partial relief, but none of them addresses every obstacle on its own. Addressing the
problem fully requires changes at the level of publication norms and evaluation criteria, not just better intentions from
authors of individual papers.
""",
    "suspicious-document00004": """\
Threshold Selection in Similarity-Based Retrieval Systems

Introduction

Many information retrieval systems, plagiarism detectors among them, ultimately reduce
a comparison between two pieces of text to a single similarity score, and then must decide
whether that score is high enough to count as a match. Choosing this threshold sounds like
a minor implementation detail, but in practice it has an outsized effect on system
behaviour, and the right way to choose it depends heavily on what kind of mistake the
system's designers are most trying to avoid.

The Trade-off Between Precision and Recall

Lowering a similarity threshold makes a system more permissive: it flags more candidate
matches, which increases the chance of catching genuine cases of overlap but also
increases the number of false positives, cases where the system reports a match that
does not actually represent meaningful similarity. Raising the threshold has the opposite
effect, reducing false positives at the cost of missing some genuine matches that happen
to fall just below the cutoff. Because real similarity scores form a continuous distribution
rather than two cleanly separated clusters, there is rarely a threshold value that achieves
zero of both error types simultaneously, and system designers are forced to choose a
point on this trade-off curve explicitly.

Domain-Dependent Calibration

A threshold tuned on one type of document collection often transfers poorly to another.
A system calibrated on long-form academic papers, where genuine topical overlap
between unrelated documents is common simply because authors discuss the same
established background material, may need a stricter threshold than a system calibrated
on short student essays, where overlap is rarer and therefore more likely to be meaningful
when it occurs. This means published threshold values from one study should rarely be
adopted unchanged in a different application without re-validation on representative data
from the new domain.

Static Thresholds versus Adaptive Thresholds

The simplest approach is to fix a single global threshold value and apply it uniformly to
every comparison. This is easy to implement and to explain, but it ignores the fact that
some documents are simply more "similar by default" than others — for example, two
short documents on a narrow technical topic will tend to share more vocabulary than two
long documents on diverse topics, independent of any copying. Adaptive approaches
instead calibrate the threshold per document or per document pair, for instance by
comparing a candidate similarity score against the distribution of similarity scores
between the suspicious document and a large random sample of unrelated documents,
and flagging only scores that are unusual relative to that local baseline rather than
unusual in some fixed, global sense.

Cascading Thresholds in Multi-Stage Pipelines

Modern retrieval pipelines frequently use more than one threshold at different stages: an
initial, deliberately loose threshold filters an enormous candidate pool down to a
manageable shortlist using a cheap method, and a second, stricter threshold (sometimes
backed by a more expensive model) makes the final decision on this shortlist. This
staged design changes how threshold errors propagate through the system. An overly
strict threshold at the first stage is particularly costly because it can discard a true match
before the more accurate second-stage method ever has a chance to evaluate it, whereas
an overly loose first-stage threshold mostly costs extra computation at the second stage
rather than lost recall.

Threshold Drift Over Time

A threshold that was well-calibrated when a system launched can become poorly
calibrated months or years later, simply because the underlying document collection has
changed. If a retrieval system is gradually populated with documents from a narrower
range of topics than it originally covered, average pairwise similarity within the collection
tends to rise, and a threshold that once separated genuine matches from coincidental
overlap may start flagging an increasing number of false positives without anyone having
changed the threshold value itself. Detecting this kind of drift requires monitoring the
distribution of similarity scores over time, not just monitoring whether the threshold value
configured in the system has changed, since the threshold can stay numerically fixed while
its practical effect shifts considerably.

Threshold Selection Under Class Imbalance

Many retrieval applications, plagiarism detection very much included, operate under
severe class imbalance: the overwhelming majority of document pairs compared by the
system are unrelated, and only a tiny fraction represent genuine matches. Under this kind
of imbalance, even a threshold with a very low false positive rate in percentage terms can
still produce more false positives than true positives in absolute terms, simply because
there are vastly more opportunities for a false positive to occur than a true positive.
Evaluating a proposed threshold purely on its false positive rate, without separately
accounting for how rare true positives are expected to be in the deployment setting, can
therefore give a misleadingly optimistic picture of how the system will behave once
deployed against a real, highly imbalanced document collection.

Reporting Thresholds Transparently

A separate, more practical concern is that published research on retrieval and detection
systems does not always report the threshold values used, or the process by which they
were chosen, in enough detail for another team to reproduce the reported results on a
different collection. Without this information, a reported precision or recall figure is hard
to interpret, since the same underlying method can produce very different precision-recall
trade-offs purely by adjusting its threshold, with no change to the method itself. Clear
reporting of threshold values, and ideally of the full precision-recall curve across a range of
threshold settings rather than a single operating point, gives readers a much more
complete picture of how a system is likely to behave under different deployment priorities.

Learned Thresholds versus Hand-Tuned Thresholds

An increasingly common alternative to manually selecting a single threshold is to learn an
appropriate decision boundary directly from labelled data, treating threshold selection as a
small supervised learning problem rather than a manual tuning exercise. This approach can
adapt the boundary to subtle interactions between features that a human tuner would be
unlikely to notice, such as a threshold that should itself depend on document length or on
the number of candidate matches already found for a given query. The cost of this
flexibility is a dependency on having enough labelled examples to fit the boundary reliably,
and a learned threshold inherits whatever biases are present in its training labels, which
can be a serious problem if the labelled dataset used to fit the threshold does not
represent the full diversity of cases the deployed system will eventually encounter. A
hand-tuned threshold, by contrast, is more transparent and easier to adjust manually when
a system's designers notice it behaving badly in a particular edge case, even though it
cannot capture the same subtle feature interactions a learned boundary can.

Threshold Selection in the Presence of Adversarial Pressure

Plagiarism detection thresholds face a complication that many other retrieval applications
do not: the population being measured has an active incentive to find inputs that fall just
below whatever threshold is in use. This adversarial pressure means a threshold that
performs well on a static, non-adversarial evaluation set can perform considerably worse
once deployed, because some fraction of the population being measured will actively
probe for the boundary and adjust their behaviour specifically to stay just under it. Systems
operating under this kind of pressure sometimes adopt a deliberately conservative
threshold, accepting a higher false positive rate than a non-adversarial setting would
justify, precisely because the alternative is a slow but steady erosion of effective recall as
the population being measured adapts to whatever threshold was originally chosen.

Threshold Selection with Multiple Similarity Signals

Many modern systems compute several different similarity signals for the same document
pair, such as a lexical overlap score, an embedding-based score, and a structural similarity
score, and must decide how to combine these signals into a single decision rather than
thresholding any one of them in isolation. The simplest combination strategy applies a
separate threshold to each signal and treats a pair as a match if any individual threshold is
exceeded, which favours recall but multiplies the opportunities for a false positive across
signals. A more conservative strategy requires several signals to simultaneously exceed
their respective thresholds, which favours precision but can miss genuine matches that
register strongly on only one signal because of how the underlying obfuscation happens to
interact with that particular detection method. Learned combination strategies, such as
fitting a small classifier on top of the raw signal values, can in principle outperform either
fixed combination rule, but they reintroduce the same training-data dependency and
distributional-drift concerns already discussed in the context of single learned thresholds.

Per-User and Per-Cohort Threshold Calibration

A further refinement worth considering is calibrating thresholds not just per document
collection but per identifiable subgroup within that collection. A plagiarism detection
system deployed across an entire university may face a meaningfully different similarity
distribution in a first-year introductory course, where students draw heavily on a small
shared set of assigned readings and therefore exhibit elevated baseline similarity to each
other's submissions purely from discussing the same material, compared with a final-year
research seminar, where topics are individually chosen and genuine overlap is rarer and
more suspicious when it occurs. Applying a single institution-wide threshold across both
contexts forces an unfavourable compromise: a threshold loose enough to catch genuine
plagiarism in the research seminar will generate an unmanageable volume of false
positives in the introductory course, while a threshold strict enough to be usable in the
introductory course risks missing real cases in the seminar setting. Per-cohort calibration
addresses this directly, at the cost of requiring enough historical data within each cohort to
calibrate reliably, which is not always available for small or newly created courses.

The Cost of Threshold Errors Is Rarely Symmetric

Much of the preceding discussion treats false positives and false negatives as comparably
costly, but in most real deployments the two error types carry very different practical
consequences, and this asymmetry should inform threshold choice directly rather than
being treated as a secondary consideration after a threshold has already been selected on
purely statistical grounds. A false positive in an academic integrity context typically results
in additional reviewer time and, if mishandled, reputational and emotional harm to a
wrongly accused student, whereas a false negative simply allows an instance of
undetected misconduct to pass through unflagged, with consequences that are diffuse and
delayed rather than immediate and personal. Many institutions, recognising this
asymmetry, deliberately bias their thresholds toward minimising false accusations even at a
measurable cost to recall, a choice that is defensible on ethical grounds but that is easy to
overlook if threshold selection is treated as a purely technical optimisation problem
divorced from the human consequences of each error type.

Communicating Threshold Decisions to Non-Technical Stakeholders

A threshold value chosen through careful empirical analysis is of limited practical use if the
people who must act on a system's output, such as academic integrity committees or
non-technical administrators, do not understand what the threshold actually represents.
A similarity score of, say, seventy percent is easy to misinterpret as a direct probability of
guilt, when it more accurately reflects how much text overlap was detected relative to
some calibration baseline that the stakeholder has no visibility into. Systems that present
raw similarity scores without context invite exactly this kind of misinterpretation, and
some institutions have responded by translating numeric scores into a small number of
qualitative bands, such as low, moderate or high concern, calibrated so that each band
corresponds to a roughly consistent likelihood of confirmed misconduct upon human
review. This translation step sacrifices some precision in exchange for making the
threshold's practical meaning legible to the people ultimately responsible for deciding
what happens to a flagged case.

Conclusion

There is no universal "correct" similarity threshold; the right value depends on the cost of
false positives relative to false negatives in the specific application, on how similarity
scores are distributed within the document collection being searched, and on whether the
threshold sits in an early filtering stage or a final decision stage of a larger pipeline. Systems
that treat threshold selection as a one-time calibration step, rather than an ongoing
empirical question revisited as the document collection changes and as class imbalance is
properly accounted for, tend to degrade quietly over time as their assumptions drift away
from the data they are actually seeing.
""",
    "suspicious-document00005": """\
The Limits of Self-Reported Academic Integrity Surveys

Introduction

A large fraction of what is known about how common plagiarism and other forms of
academic dishonesty are among students comes from self-report surveys: students are
asked, often anonymously, whether they have engaged in particular behaviours over some
time period. This essay considers how reliable this method actually is, and what
alternative or complementary approaches exist.

Social Desirability Bias

Self-report surveys about a behaviour widely understood to be wrong are vulnerable to
social desirability bias: respondents tend to under-report behaviours they expect to be
judged for, even when a survey is anonymous, simply because admitting to the behaviour
on paper still feels uncomfortable. Researchers have tried to mitigate this with various
techniques, such as randomised response methods that introduce statistical noise into
individual answers while still allowing accurate estimates of the prevalence across a group,
but these techniques add complexity and are not used in most plagiarism surveys, which
typically rely on simpler direct questioning.

Definitional Ambiguity

A second problem is that students and researchers do not always share the same
definition of plagiarism. A student who paraphrases a textbook definition without citation
may not consider this "real" plagiarism in the way that copying an entire essay would be,
even though many academic integrity policies would classify both as violations. Surveys
that ask broad questions like "have you ever plagiarised" will therefore undercount
behaviours that respondents do not personally categorise as plagiarism, while surveys
that list specific behaviours in detail tend to report higher prevalence rates simply because
they prompt recognition of behaviours the respondent would not have spontaneously
labelled as plagiarism.

Recall and Time-Frame Effects

Asking about behaviour over a long time frame, such as "during your entire degree",
introduces recall error: minor incidents are forgotten, while major or recent incidents are
overrepresented in what respondents report. Shorter, more frequent surveys covering a
single term or semester tend to produce more accurate prevalence estimates but are
logistically harder to run repeatedly across an entire student population, and response
rates often decline with each additional survey wave.

Triangulating with Behavioural and Detection Data

Because of these limitations, some researchers have begun cross-referencing survey
results against data from plagiarism detection systems used by the same institution, such
as the rate at which submitted assignments are flagged for high similarity scores. This
triangulation is useful but imperfect in the other direction: detection software has its own
false positive and false negative rates, and flagged similarity does not always correspond
to confirmed misconduct after human review, so detection rates are not a clean ground
truth against which to validate survey responses either.

Cultural and Institutional Variation

Comparing self-reported plagiarism rates across countries or institutions is particularly
fraught, because both the social stigma attached to admitting dishonest behaviour and the
local definition of what counts as plagiarism vary substantially. A cross-national
comparison that does not account for these differences risks concluding that plagiarism
is more common in one country than another, when the actual difference may largely
reflect differing willingness to report it rather than differing underlying behaviour.

Anonymity Guarantees and Respondent Trust

The strength of an anonymity guarantee offered to survey respondents has a measurable
effect on response honesty, but stronger guarantees often come at the cost of weaker
follow-up data. A survey that collects no identifying information whatsoever maximises
respondent comfort but prevents researchers from linking survey responses to other data
sources, such as actual grades or detection software flags, for the same individual. A
survey that retains a pseudonymous identifier allows this kind of linkage but introduces a
trust question: respondents must believe the institution's promise that the identifier will
not be used punitively, and any past instance of an institution breaking such a promise,
even in an unrelated context, can suppress honest responses across an entire future survey
population. This tension between data utility and respondent trust is rarely resolved
cleanly and tends to be negotiated differently by each institution's ethics review process.

Survey Fatigue and Population-Level Bias

Repeated surveying of the same student population introduces another, more subtle bias:
students with strong opinions about academic integrity, in either direction, may be more
likely to keep responding to successive survey waves than students who are indifferent to
the topic, gradually shifting the respondent pool away from being representative of the
full population. This selection effect can produce systematic drift in measured prevalence
over a multi-year study, even if no actual change in underlying student behaviour has
occurred, simply because the people willing to keep answering the survey are not a
constant, representative cross-section of the student body across all of the survey's waves.

Implications for Policy Design

Because of all the limitations discussed here, policies built on a single point-estimate
prevalence figure from a single self-report survey are standing on shakier ground than the
precision of that figure might suggest. Institutions making significant decisions, such as
allocating budget toward a particular integrity intervention or judging whether an existing
intervention has succeeded, are better served by tracking trends across repeated surveys
using a consistent methodology, rather than treating any single survey's point estimate as
a precise, comparable-across-context measurement of true prevalence.

Question Framing and Order Effects

The specific wording and ordering of survey items can shift reported prevalence by a
surprisingly large margin, independent of any underlying change in actual behaviour.
Asking respondents about minor, widely tolerated infractions before asking about more
serious ones tends to establish a permissive frame that can inflate honest reporting on the
more serious items that follow, since respondents have already implicitly acknowledged
engaging in some degree of academic shortcut-taking. The reverse ordering, leading with
the most serious infractions, can suppress reporting on subsequent milder items because
respondents have just been primed to think about plagiarism in its most severe form and
may not recognise milder behaviours as belonging to the same category. Because different
studies in the literature use different question orderings without always reporting this
methodological detail prominently, cross-study comparisons of prevalence figures are less
reliable than a simple reading of the reported numbers would suggest.

Incentive-Compatible Survey Design

A smaller body of methodological work has explored incentive-compatible elicitation
techniques borrowed from experimental economics, in which respondents are rewarded
for answers that turn out to be verifiably accurate rather than simply asked to self-report
honestly. These techniques are difficult to apply directly to a question like "have you ever
plagiarised," since there is rarely an objective, independently verifiable ground truth
against which a given respondent's honesty could be rewarded after the fact. Researchers
have instead applied incentive-compatible framings to related but more verifiable
questions, such as predicting what fraction of one's peers would admit to a given
behaviour, on the theory that people are more willing to make an honest prediction about
others' behaviour than to confess their own, and that this prediction can be combined with
modelling assumptions to back out an estimate of true individual-level prevalence.

Mode Effects: Paper, Online and In-Person Administration

The medium through which a survey is administered also measurably affects response
honesty. Online surveys completed privately and without time pressure tend to elicit more
candid responses to sensitive questions than in-person interviews, where a respondent
must answer in front of another person and may feel a more immediate, embodied form of
social pressure even when reassured the interview is confidential. Paper surveys
administered in a classroom setting occupy an intermediate position: more private than an
interview, but still completed in the physical presence of peers and an instructor, which
can subtly affect willingness to write down an honest answer to a sensitive question even
when responses are collected anonymously and out of view. Comparing prevalence
estimates across studies that used different administration modes without adjusting for
this factor risks attributing a difference to substantive causes, such as a genuine change in
student behaviour over time, when a portion of the difference may simply reflect a switch
from one administration mode to another between study waves.

Sampling Frame and Selection into the Study

Even before a respondent decides how honestly to answer a given question, the way a
study recruits its sample shapes what the resulting prevalence estimate actually
represents. Surveys distributed through a course mailing list capture only students still
enrolled and engaged enough to read course communications, systematically excluding
students who have disengaged, transferred, or withdrawn, some of whom may have left
partly because of unresolved integrity proceedings that are precisely the kind of case a
prevalence study would want to capture. Voluntary participation compounds this further: a
student who has engaged in serious misconduct may be less willing to volunteer for a
survey about academic integrity at all, regardless of how anonymous the instrument is,
producing a sample that under-represents exactly the population a researcher most wants
to characterise. Estimating the size and direction of this selection bias is difficult without
external data on the population that did not respond, which is rarely available, so most
published prevalence figures should be read as conditional on study participation rather
than as unconditional population estimates.

Longitudinal Designs and Within-Person Change

A smaller number of studies track the same individuals over multiple time points rather
than surveying a fresh cross-section each wave, and these longitudinal designs offer a
genuinely different kind of evidence than repeated cross-sectional surveys can provide.
Where a cross-sectional comparison across years can only describe how the aggregate
prevalence figure has shifted, a longitudinal design can in principle show whether
individual students who admitted to misconduct early in their studies become more or
less likely to report similar behaviour later, offering insight into whether academic
integrity behaviour is a relatively stable individual trait or a context-dependent response
to specific course pressures that fluctuates considerably within the same person across
different terms. Longitudinal designs are considerably more expensive to run, since they
require maintaining contact with the same cohort over years and contending with
attrition as participants leave the institution, which is part of why cross-sectional surveys
remain far more common in this literature despite the additional insight a longitudinal
design could offer.

Comparing Self-Report Data Against Qualitative Interview Evidence

A smaller complementary literature uses in-depth qualitative interviews, rather than
structured surveys, to understand how students themselves reason about the boundary
between acceptable collaboration and academic dishonesty. These interviews often reveal
that students apply considerably more nuanced, context-dependent distinctions than a
fixed survey instrument can capture, for instance distinguishing sharply between
discussing a problem set's approach with a classmate and directly copying that classmate's
written solution, a distinction that a simple yes-or-no survey item about "working with
others on assignments" collapses entirely. Qualitative work of this kind cannot produce a
defensible prevalence estimate, since interview samples are necessarily small and rarely
representative, but it serves a different and complementary function: revealing the
internal logic respondents are actually applying when they answer a structured survey
item, which can help researchers design better-calibrated survey instruments for future
quantitative work rather than replacing such instruments outright.

Disciplinary Differences in Reported Prevalence

Prevalence estimates also vary systematically by academic discipline, and this variation is
itself ambiguous between several competing explanations. Higher reported rates in some
technical disciplines might reflect a genuinely higher underlying rate of misconduct, driven
perhaps by larger class sizes that make detection feel less likely, or it might instead reflect
disciplinary differences in how broadly students interpret what counts as plagiarism in the
first place, with some disciplines treating close paraphrasing of a textbook far more
liberally than others. Untangling a true behavioural difference from a definitional or
detection-related artefact requires comparing disciplines using a single, carefully
standardised survey instrument administered under comparable conditions, which is rarer
in the literature than discipline-specific surveys conducted by researchers situated within
a single field who are primarily interested in their own discipline's local picture rather
than in a directly comparable cross-disciplinary estimate.

Conclusion

Self-report surveys remain a useful and widely used tool for studying academic integrity,
but their numbers should be read as lower-bound estimates shaped by social desirability,
definitional ambiguity and recall effects, rather than as precise measurements of true
prevalence. Combining survey data with detection system data and qualitative interviews
gives a more complete, if still imperfect, picture than any single method alone.
""",
}


def build_clean_docs() -> None:
    for doc_id, text in CLEAN_DOCS.items():
        (SUSP_DIR / f"{doc_id}.txt").write_text(text, encoding="utf-8")
        (GT_DIR / f"{doc_id}.xml").write_text(
            f'<document reference="{doc_id}.txt">\n</document>\n', encoding="utf-8"
        )
        print(f"{doc_id}: {len(text)} chars, 0 plagiarised spans (clean)")


# ---------------------------------------------------------------------------
# 5 PLAGIARISED documents (00011-00020 — spaced per README; we use 00011-00015 as the 5 IDs)
# ---------------------------------------------------------------------------

def build_doc_00011() -> None:
    """Verbatim — direct copy-paste of paragraphs from a single source."""
    src_id = "source-document00019"
    src = source_text(src_id)

    p1 = ("ChatGPT is a variant of the GPT-3 (Generative Pre-trained Transformer 3, Brown et al., 2020) artificial  "
          "intelligence  language  model  developed  by  OpenAI.  It  is  specifically  designed  to "
          "generate human-like text in a conversational style, and was introduced in 2021. It has received significant  "
          "attention  in  the  media  and  tech  industry.  GPT-3  is  based  on  the  Transformer "
          "architecture, which was introduced in a paper by (Vaswani et al., 2017) and has since become "
          "widely used in natural language processing tasks. GPT-3 is notable for its size, with 175 billion "
          "parameters, making it one of the largest language models currently available. It is notable for its  "
          "ability  to  perform  a  wide  range  of  language  tasks,  including  translation,  summarisation, "
          "question answering, and text generation, with little or no task-specific training.")
    p1_off = find_offset(src, p1, src_id)

    p2 = ("One of the main advantages of artificial intelligence language model is that they provide a  "
          "platform  for  asynchronous  communication.  This  feature  has  been  found  to  increase "
          "student  engagement  and  collaboration,  as  it  allows  students  to  post  questions  and "
          "discuss topics without having to be present at the same time (Li &amp; Xing, 2021). Another "
          "advantage of chatAPIs is that they can be used to facilitate collaboration among students. For "
          "instance, chatAPIs can be used to create student groups, allowing students to work together  "
          "on  projects  and  assignments  (Lewis,  2022).  Finally,  chatAPIs  can  be  used  to "
          "enable remote learning. This is especially useful for students who are unable to attend "
          "classes due to physical or mental health issues (Barber et al., 2021).")
    p2_off = find_offset(src, p2, src_id)

    b = DocBuilder()
    b.add(
        "Large Language Models in Higher Education: A Case Study Review\n\n"
        "Introduction\n\n"
        "Generative AI tools have moved from research curiosities to widely used consumer products in "
        "the span of only a couple of years. Nowhere has this shift been felt more acutely than in higher "
        "education, where instructors and students alike are still working out what these tools mean for "
        "how coursework gets written, assessed, and trusted. This report looks at one of the most discussed "
        "tools in this space and summarises its background before turning to its classroom implications.\n\n"
        "Background on the Technology\n\n"
    )
    b.add_plagiarised(p1, src_id, p1_off, "verbatim", "none")
    b.add(
        "\n\nThis combination of scale and general-purpose language ability is what makes the tool relevant "
        "well beyond its original research context, and explains why so much attention has shifted toward "
        "its use, and potential misuse, inside classrooms.\n\n"
        "Opportunities for Student Engagement\n\n"
    )
    b.add_plagiarised(p2, src_id, p2_off, "verbatim", "none")
    b.add(
        "\n\nTaken together, these capabilities suggest that the technology can lower some of the practical "
        "barriers that have historically limited participation in collaborative coursework, particularly for "
        "students juggling external commitments.\n\n"
        "Concerns Raised by Educators\n\n"
        "Despite these advantages, instructors have raised consistent concerns about how easily such tools "
        "can be used to produce submitted work that does not reflect a student's own understanding. Unlike "
        "earlier forms of academic dishonesty, text generated this way is original in the sense that it does "
        "not match any existing document word-for-word, which makes traditional similarity-based detection "
        "tools far less effective at flagging it. Institutions have responded with a mix of policy "
        "statements, revised assessment design, and pilot programmes for AI-output detectors, though none "
        "of these responses has yet produced a fully reliable solution.\n\n"
        "Detection as a Moving Target\n\n"
        "Every time a new generation of language model becomes available, the detection tools built around "
        "the previous generation lose some of their effectiveness. Classifiers trained to recognise the "
        "statistical fingerprints of one model's output — its preferred sentence lengths, its vocabulary "
        "distribution, its tendency to hedge claims with particular phrasing — generalise poorly to a newer "
        "model trained with different objectives or fine-tuned on different data. This creates an arms-race "
        "dynamic that mirrors, in miniature, the older arms race between plagiarists using paraphrasing tools "
        "and the similarity detectors built to catch them. The practical consequence for an institution is "
        "that any AI-detection tool purchased today should be expected to degrade in accuracy within a year "
        "or two, simply because the underlying generative models it was calibrated against will have moved "
        "on.\n\n"
        "Policy Responses Across Institutions\n\n"
        "Universities have not converged on a single policy stance. Some have banned the use of generative "
        "AI tools for any graded work, treating any detected use as a straightforward integrity violation "
        "comparable to copying from another student. Others have taken the opposite approach, explicitly "
        "permitting AI assistance for certain types of assignment provided the student discloses which "
        "sections were AI-generated and can explain the reasoning behind the final submitted answer. A third "
        "group has tried to sidestep the detection problem entirely by redesigning assessments toward formats "
        "that are harder for a language model to complete convincingly on its own, such as oral examinations, "
        "in-class problem solving under time pressure, or assignments that require referencing very recent, "
        "course-specific material the model is unlikely to have been trained on. Each of these approaches "
        "carries its own costs: outright bans are difficult to enforce consistently, permissive disclosure "
        "policies rely on student honesty about exactly the same kind of disclosure plagiarism policies "
        "already struggle to enforce, and assessment redesign is labour-intensive for instructors and not "
        "always feasible for large courses with hundreds of students.\n\n"
        "Equity Considerations\n\n"
        "An underexplored dimension of this debate is unequal access. Some generative AI tools are offered "
        "free of charge with usage limits, while more capable versions sit behind a subscription paywall. "
        "Students who can afford premium access to a more capable model may produce work that is both higher "
        "quality and harder to detect than students relying on free tiers, introducing a new axis of "
        "inequality that traditional academic integrity policy was not designed to address. Likewise, "
        "students who are non-native speakers of the language of instruction may have legitimate, "
        "integrity-compliant reasons to use AI tools for grammar and phrasing assistance, and a blanket "
        "detection-and-punishment policy risks disproportionately flagging this group even when no '"
        "dishonesty has occurred.\n\n"
        "Faculty Workload and the Limits of Manual Vigilance\n\n"
        "Even where an institution chooses not to rely on automated AI-output detectors at all, the burden of "
        "noticing suspicious submissions does not disappear; it simply shifts back onto individual instructors, "
        "who must now read student work with a heightened, largely informal suspicion that was not part of "
        "grading practice a few years ago. For instructors teaching large courses with hundreds of submissions "
        "per assignment, this informal vigilance is necessarily inconsistent: a marker grading the fortieth "
        "essay in a sitting is less likely to notice a subtly generic, AI-typical writing style than one grading "
        "the third, simply due to ordinary fatigue rather than any difference in the underlying writing. This "
        "unevenness means that even a policy nominally enforced through human judgement alone produces "
        "outcomes that depend heavily on logistical factors like grading order and class size, which sits "
        "uncomfortably alongside the principle that academic integrity enforcement should be applied "
        "consistently across a cohort regardless of when in the grading queue a particular submission happens "
        "to fall.\n\n"
        "Disciplinary Variation in Adoption and Concern\n\n"
        "The intensity of concern about generative AI differs considerably across academic disciplines, "
        "tracking how central extended written prose is to a discipline's normal assessment practice. "
        "Humanities departments, where essays and analytical writing constitute the primary evidence of "
        "student learning, have generally voiced the loudest concern, since a generated essay can satisfy the "
        "surface requirements of an assignment without the student having engaged with the assigned material "
        "at all. Quantitative disciplines that assess primarily through problem sets, derivations or coding "
        "exercises have engaged with the technology differently, often treating it as a potential aid for "
        "explanation and debugging rather than as a primary threat to assessment validity, partly because "
        "current models remain comparatively less reliable at producing fully correct, novel quantitative "
        "derivations than at producing fluent, plausible-sounding prose. This disciplinary asymmetry means "
        "that an institution-wide policy calibrated to the humanities case risks being either over-restrictive "
        "or simply irrelevant when applied unmodified to a quantitative department's actual assessment "
        "practices.\n\n"
        "Longitudinal Uncertainty About Learning Outcomes\n\n"
        "A separate and still largely unresolved question concerns what habitual reliance on these tools does "
        "to a student's underlying skill development over the multi-year span of a degree, independent of "
        "whether any individual use counts as a integrity violation under current policy. Writing is widely "
        "understood among educators not merely as a way of demonstrating already-formed thoughts but as a "
        "process through which thinking itself is clarified and tested, and a student who routinely offloads "
        "the drafting process to a generative tool may be foregoing exactly the cognitive work that produces "
        "durable improvement in reasoning and expression, even in cases where the final submitted work would "
        "not be flagged as a policy violation by any current detection method. Measuring this kind of slow, "
        "cumulative skill effect is methodologically difficult, since it requires tracking the same students "
        "across years rather than evaluating a single assignment in isolation, and the technology itself has "
        "not been available long enough for any such longitudinal study to have run its full course.\n\n"
        "Institutional Investment in Detection Infrastructure\n\n"
        "Some universities have responded to this uncertainty by purchasing institution-wide licences for "
        "commercial AI-detection services and integrating them directly into existing plagiarism-checking "
        "workflows, treating an AI-likelihood score as simply one more piece of evidence alongside a "
        "traditional similarity score. This integration is administratively convenient, since it slots into "
        "infrastructure instructors already use, but it risks conferring an unwarranted sense of reliability "
        "on a class of detector that is, by the vendors' own published accuracy figures, considerably less "
        "dependable than the similarity-matching technology it sits alongside. An instructor accustomed to "
        "treating a high traditional similarity score as fairly strong evidence may not adequately discount an "
        "AI-likelihood score that, despite being presented in a similar numeric format, carries a meaningfully "
        "higher rate of misclassification, particularly against text written by non-native speakers or text "
        "covering a narrow technical topic with limited natural vocabulary variation.\n\n"
        "Comparative Policy Snapshots\n\n"
        "Looking across the small number of published institutional case studies available so far, a rough "
        "pattern emerges in how policy stance correlates with prior institutional culture around academic "
        "integrity more broadly. Institutions that already operated comparatively strict, zero-tolerance "
        "integrity regimes before generative AI tools became widely available have tended to extend that same "
        "strictness to AI use by default, treating undisclosed use as equivalent in severity to other forms "
        "of unauthorised assistance. Institutions with a more developmental, educative approach to integrity "
        "violations, oriented toward correcting behaviour rather than punishing it, have tended to extend that "
        "same developmental framing to AI use, treating a first undisclosed instance as an occasion for "
        "guidance on appropriate use rather than an automatic disciplinary referral. This continuity with "
        "prior institutional culture suggests that policy toward generative AI is, in practice, less a fresh "
        "decision made on the technology's own merits and more an extension of whatever integrity philosophy "
        "an institution had already settled into before the technology arrived.\n\n"
        "The Student Perspective on Disclosure Requirements\n\n"
        "Disclosure-based policies, which permit AI assistance provided the student declares which portions "
        "of a submission were AI-assisted, place a burden of self-report on the student that is structurally "
        "similar to the burden self-report academic integrity surveys place on respondents more broadly, and "
        "inherits some of the same reliability concerns. A student uncertain whether a particular use, such as "
        "asking a model to suggest alternative phrasing for an already-drafted paragraph, counts as the kind "
        "of assistance that must be disclosed under a given policy faces genuine ambiguity that the policy "
        "itself may not resolve clearly, and in the absence of clear guidance, students may default toward "
        "under-disclosure simply to avoid the risk of being penalised for an ambiguous case. Institutions "
        "adopting disclosure-based policies have generally found that clear, example-rich guidance describing "
        "specific permitted and prohibited use cases produces more consistent and more honest disclosure "
        "behaviour than a single abstract policy statement left for students to interpret on their own.\n\n"
        "Conclusion\n\n"
        "The technology under discussion illustrates a broader pattern: tools developed primarily for "
        "general-purpose language tasks tend to diffuse into education faster than institutional policy can "
        "adapt to them. Understanding both the underlying architecture and the concrete classroom use cases "
        "is a prerequisite for designing assessment practices that remain meaningful in this new environment, "
        "and the policy responses surveyed here suggest that no single approach — banning, permitting with "
        "disclosure, or redesigning assessment — is likely to be sufficient on its own across every "
        "institutional context.\n"
    )
    write_doc("suspicious-document00006", b)


def build_doc_00012() -> None:
    """Verbatim — direct copy-paste, multiple sources."""
    src1_id = "source-document00018"
    src1 = source_text(src1_id)
    p1 = ("Social, technological and information systems can often be described in terms of complex networks "
          "that have a topology of interconnected nodes combining organization and randomness [1, 2]. The "
          "typical size of large networks such as social network services, mobile phone networks or the web "
          "is now counted in millions, if not billions, of nodes and these scales demand new methods to "
          "retrieve comprehensive information from their structure. A promising approach consists in "
          "decomposing the networks into sub-units or communities, which are sets of highly interconnected "
          "nodes [3]. The identification of these communities is of crucial importance as they may help to "
          "uncover a priori unknown functional modules such as topics in information networks or "
          "cyber-communities in social networks. Moreover, the resulting meta-network, whose nodes are the "
          "communities, may then be used to visualize the original network structure.\n\n"
          "The problem of community detection requires the partition of a network into communities of "
          "densely connected nodes, with the nodes belonging to different communities being only sparsely "
          "connected. Precise formulations of this optimization problem are known to be computationally "
          "intractable. Several algorithms have therefore been proposed to find reasonably good partitions "
          "in a reasonably fast way. This search for fast algorithms has attracted much interest in recent "
          "years due to the increasing availability of large network datasets and the impact of networks on "
          "everyday life. One can distinguish several types of community detection algorithms: divisive "
          "algorithms detect inter-community links and remove them from the network [4]-[6], agglomerative "
          "algorithms merge similar nodes/communities recursively [7] and optimization methods are based on "
          "the maximization of an objective function [8]-[10].")
    p1_off = find_offset(src1, p1, src1_id)

    src2_id = "source-document00016"
    src2 = source_text(src2_id)
    p2 = ("Support  Vector  Machines  have  shown  their  capacities  in "
          "pattern recognition.  The aim of SVM classification method is to find the best hyper -plane "
          "separating relevant and irrelevant vectors  maximizing  the  size  of  the  margin  (between  both "
          "classes).    Initial  method  assumes  that  relevant  and  irrelevant "
          "vectors  are  linearly  separable  [6].    The  SVM  separate  the "
          "whole image database into two classes.  The two classes are also including the unlabelled images "
          "with two types they are relevant and  irrelevant unlabelled images. The  relevant unlabelled image "
          "is related to the relevant labelled images in the  image database.  In similar way  the irrelevant "
          "unlabelled image  is  related  to  the  irrelevant  labelled  images  in  the database.  This SVM "
          "is also classifying the unlabelled images in accuracy manner.\n\n"
          "## C. Relevance Feedback\n\n"
          "As  image  databases  usually  contain  unlabelled  images, these can be exploited to help "
          "supervised learning by asking the  user  to  label  them.  The  goal  is  naturally  to  minimize "
          "asking  help  from  the  user.    The  relationships  of  all  the  data points in the feature "
          "space using a manifold ranking algorithm and constructs a weighted graph that contains all "
          "images; the ranking scores of labelled examples are iteratively propagated to  nearby  labelled  "
          "and  unlabelled  images[2].  The  Relevance Feedback is using the likelihood functions used to "
          "relate the certain data points in the image class.  These function is define the relevancy of "
          "each data point to the user given query.  This is most useful to give rank to unlabelled images "
          "in the image database  [9,  49].    The  learners  are  trained  with  labelled  and unlabelled  "
          "images.")
    p2_off = find_offset(src2, p2, src2_id)

    b = DocBuilder()
    b.add(
        "Network Communities and Classification: Two Methods Compared\n\n"
        "Introduction\n\n"
        "This short review brings together two lines of work that, on the surface, look unrelated: detecting "
        "community structure in large networks, and classifying images by relevance using supervised "
        "learning. Both, however, share a common underlying need to partition a large search space into "
        "more manageable, internally coherent groups before any further processing can happen.\n\n"
        "Community Structure in Networks\n\n"
    )
    b.add_plagiarised(p1, src1_id, p1_off, "verbatim", "none")
    b.add(
        "\n\nThe identification of such communities is treated as a prerequisite step in many network "
        "analysis pipelines, since most downstream questions about a network's structure are easier to "
        "answer once a partition into communities is available.\n\n"
        "Classification via Support Vector Machines\n\n"
    )
    b.add_plagiarised(p2, src2_id, p2_off, "verbatim", "none")
    b.add(
        "\n\nBoth techniques illustrate a recurring theme in large-scale data analysis: a hard global "
        "optimisation problem is replaced with a tractable local heuristic that performs well in practice "
        "even though it offers no guarantee of finding a global optimum.\n\n"
        "Scalability Considerations\n\n"
        "Scalability concerns manifest differently for the two problems. Community detection algorithms are "
        "typically applied to a single very large graph and are bottlenecked by memory and by how the graph "
        "is partitioned across machines when it does not fit on one node. SVM-based classification, by "
        "contrast, is usually bottlenecked by the number of training examples and the dimensionality of the "
        "feature space, since the core optimisation involves a quadratic programming problem whose "
        "complexity grows faster than linearly with the size of the training set in the general case. This "
        "is why practitioners working with very large labelled datasets often turn to linear SVM variants or "
        "stochastic gradient methods that trade some classification accuracy for the ability to train on "
        "millions of examples in a reasonable amount of time.\n\n"
        "Evaluation Metrics\n\n"
        "Comparing methods within either field also requires care about which evaluation metric is "
        "appropriate. For community detection, modularity is the most widely used metric, but it has known "
        "limitations, including a resolution limit that can prevent it from detecting small communities "
        "embedded within a much larger network. For SVM classification, accuracy alone is rarely sufficient "
        "when classes are imbalanced, which is the normal case in image retrieval where relevant images are a "
        "small minority of a large database; precision, recall and area under the ROC curve are typically "
        "reported alongside or instead of raw accuracy in that setting.\n\n"
        "Practical Implementation Notes\n\n"
        "In production systems, both techniques are rarely deployed in their textbook form. Community "
        "detection implementations commonly use the two-phase local-optimisation-then-aggregation pattern "
        "because it scales to networks with hundreds of millions of edges on commodity hardware, while exact "
        "or near-exact modularity optimisation would not. SVM implementations for large-scale image retrieval "
        "commonly use approximate kernel methods or convert to a linear formulation in a transformed feature "
        "space, since exact kernel SVMs scale quadratically or worse with the number of training examples and "
        "become impractical once datasets reach the size typical of modern image collections.\n\n"
        "Parameter Sensitivity and Model Selection\n\n"
        "Both families of method expose tunable parameters whose effect on output quality is not always "
        "intuitive to a practitioner approaching the method for the first time. Community detection algorithms "
        "built around modularity optimisation are comparatively parameter-light in their basic form, but "
        "variants that introduce a resolution parameter to control the typical size of detected communities "
        "require a principled strategy for choosing that parameter, since setting it incorrectly can either "
        "merge genuinely distinct communities into one or fragment a single coherent community into several "
        "spurious pieces. SVM classifiers expose a regularisation parameter controlling the trade-off between "
        "margin width and training error, plus, for non-linear kernels, additional parameters governing the "
        "kernel's shape, and poor choices here manifest as either underfitting, where the classifier fails to "
        "separate even the training data well, or overfitting, where the classifier achieves excellent "
        "training accuracy but generalises poorly to unseen examples. In both cases, practitioners typically "
        "resolve parameter selection through cross-validation, holding out a portion of labelled data to "
        "evaluate candidate parameter settings before committing to one for the full dataset.\n\n"
        "Handling Dynamic, Evolving Inputs\n\n"
        "A further point of contrast emerges once the underlying data is no longer static. Real social and "
        "communication networks evolve continuously, with edges and nodes appearing and disappearing as "
        "relationships form and dissolve, and a community structure computed once on a snapshot can become "
        "stale within a relatively short window if the network is changing quickly. Dynamic community "
        "detection methods address this by incrementally updating a previously computed partition as new "
        "edges arrive, rather than recomputing the entire partition from scratch after every change, trading "
        "some accuracy relative to a full recomputation for a substantial reduction in computational cost. "
        "Classification systems built on SVMs face an analogous problem when the underlying data distribution "
        "drifts over time, for instance as new categories of relevant image appear in a retrieval system's "
        "query stream that were not represented in the original training set; the standard response is "
        "periodic retraining on a sliding window of recent labelled examples, which shares the same underlying "
        "logic of trading some staleness for tractable computational cost.\n\n"
        "Interpretability of the Resulting Output\n\n"
        "A final point of comparison concerns how easily a non-specialist can interpret what either method "
        "has actually produced. A community detection partition can usually be inspected directly by a domain "
        "expert, who can examine the members of a given community and judge whether the grouping makes "
        "substantive sense given outside knowledge of the network's subject matter, which is part of why "
        "community detection is often used as an exploratory tool even outside fully automated pipelines. An "
        "SVM's decision boundary, by contrast, is considerably harder to inspect directly once the feature "
        "space has more than a handful of dimensions or once a non-linear kernel has been applied, since the "
        "boundary no longer corresponds to any single human-interpretable geometric object in the original "
        "feature space. This interpretability gap matters operationally: a domain expert can sanity-check a "
        "community detection result fairly quickly, while validating an SVM classifier's behaviour typically "
        "requires examining its predictions on a curated set of test cases rather than inspecting the model "
        "itself directly.\n\n"
        "Hybrid Approaches Combining Both Techniques\n\n"
        "A smaller body of work has explored combining community detection and SVM-style classification "
        "directly within a single pipeline, rather than treating them as alternative solutions to separate "
        "problems. One natural combination uses community detection as a feature-engineering step ahead of "
        "classification: a node's community membership, or summary statistics describing its local community "
        "such as community size or internal density, can be fed into a classifier as an additional feature "
        "alongside more conventional attributes, on the theory that a node's position within the broader "
        "community structure of a network carries predictive information that purely local features miss "
        "entirely. Conversely, a classifier's output can itself be used to weight or filter the edges of a "
        "network before community detection is applied, for instance down-weighting edges between nodes the "
        "classifier judges unlikely to belong to the same underlying group, which can sharpen the community "
        "structure the subsequent detection algorithm recovers. Neither direction of combination has become "
        "dominant in practice, and the choice between them tends to depend on which of the two signals, "
        "network structure or node-level features, is judged more reliable for the specific application at "
        "hand.\n\n"
        "Robustness to Noisy or Adversarially Perturbed Input\n\n"
        "Both families of method also differ in how they respond to noise or deliberate adversarial "
        "perturbation of their input. Community detection algorithms are known to be sensitive to a small "
        "number of spurious edges added between otherwise distinct communities, since even a handful of "
        "cross-community edges can be enough to merge two communities that a noise-free version of the same "
        "network would have kept clearly separated, which has motivated research into robust variants that "
        "explicitly model and discount likely noise edges rather than treating every recorded edge as equally "
        "reliable. SVM classifiers exhibit an analogous vulnerability to adversarially perturbed input "
        "features, where a small, carefully chosen perturbation to a feature vector, often imperceptible to a "
        "human inspecting the underlying image or document, can flip the classifier's decision despite the "
        "true underlying content being unchanged. This shared vulnerability to small, structured perturbations "
        "is not coincidental: both methods make decisions based on thresholds applied to continuous scores "
        "derived from the input, and small input changes that happen to be aligned with the right direction "
        "relative to that threshold can disproportionately affect the final discrete decision even when they "
        "barely affect the underlying continuous score.\n\n"
        "Historical Development and Convergent Timing\n\n"
        "It is worth noting that both techniques reached broad practical maturity within a roughly similar "
        "window, even though they originated in different research communities and were motivated by "
        "different applied problems. Modularity-based community detection methods became practical for "
        "networks with millions of nodes once efficient two-phase local-optimisation algorithms were "
        "published, after which the technique diffused quickly into social network analysis, biology and "
        "computational social science as a default tool for exploring previously unanalysed large graphs. "
        "Support vector machines, although proposed somewhat earlier in their basic form, only became "
        "practical for the scale of feature data typical in modern image and document collections once "
        "efficient training algorithms and, later, effective kernel approximation methods made it feasible to "
        "train on millions of examples without the prohibitive cost of an exact quadratic program. This "
        "convergent timing is partly coincidental and partly a reflection of the same broader trend: as "
        "datasets across many fields grew rapidly in size, methods that had been theoretically known for "
        "years became practically deployable only once someone solved the specific scalability bottleneck "
        "standing between the elegant formulation and a usable implementation at realistic data volumes.\n\n"
        "Lessons for Future Large-Scale Methods\n\n"
        "The shared history of these two techniques offers a more general lesson for evaluating any newly "
        "proposed method aimed at large-scale data analysis: theoretical elegance and provable optimality "
        "guarantees matter less in practice than whether a method admits an efficient, even if heuristic, "
        "implementation that scales to the data volumes a field actually needs to process. Both modularity "
        "optimisation and SVM training were, in their exact forms, computationally intractable at realistic "
        "scale, and both owe their widespread practical adoption to heuristic relaxations that traded "
        "provable optimality for tractability. This pattern is likely to recur as new large-scale analysis "
        "problems are formulated in the future, and it suggests that evaluating a newly proposed method "
        "purely on the theoretical properties of its exact formulation, without also asking whether a "
        "practical heuristic relaxation exists and how well that relaxation performs empirically, gives an "
        "incomplete picture of whether the method will ever see real adoption outside a research paper.\n\n"
        "Open Problems Shared by Both Fields\n\n"
        "Several open research questions recur in both literatures in only lightly disguised form, which is "
        "itself evidence that the underlying problems are more closely related than the separate vocabularies "
        "of the two fields might suggest. Determining the right number of communities to report, without "
        "relying on a user-supplied parameter, mirrors the long-standing difficulty of choosing the right "
        "number of classes or the right decision threshold in a classification setting without relying on an "
        "arbitrarily chosen cutoff. Handling networks or datasets that change incrementally over time without "
        "recomputing an expensive result from scratch is a shared concern across both fields, as already "
        "discussed, and remains only partially solved in either one. And both fields continue to wrestle with "
        "the gap between an objective function that is mathematically convenient to optimise, such as "
        "modularity or the SVM margin, and the actual downstream notion of quality a practitioner ultimately "
        "cares about, which the convenient objective only approximates and which can diverge from it in "
        "specific edge cases that matter a great deal in practice even though they are rare in any single "
        "evaluation benchmark.\n\n"
        "Conclusion\n\n"
        "Although community detection and SVM-based classification were developed for different purposes and "
        "in different research communities, comparing them side by side highlights how often large-scale "
        "data analysis reduces to the same underlying problem of finding a good, fast, approximate partition "
        "of a large space, and how the practical engineering compromises made to achieve that scalability "
        "tend to look similar even across fields that otherwise share little methodological overlap.\n"
    )
    write_doc("suspicious-document00007", b)


def build_doc_00013() -> None:
    """Near-verbatim — minor word substitutions, same structure."""
    src_id = "source-document00029"
    src = source_text(src_id)
    p1 = ("Plagiarism detection refers to techniques, tools and methods used for automated detection  of  "
          "plagiarism,  since  manual  detection  becomes  infeasible  with  large  amounts  of "
          "information.")
    p1_off = find_offset(src, p1, src_id)
    p1_mod = ("Plagiarism detection refers to the techniques, tools and methods used for automatic detection of "
              "plagiarism, given that manual review becomes impractical with large volumes of information.")

    marker = "Researchers  have  identified  paraphrase  plagiarism  in  surveys  ranging  over  two  decades "
    start_idx = src.find(marker)
    if start_idx == -1:
        raise ValueError("marker not found in source-document00029")
    end_idx = src.find(").", start_idx) + 2
    p2 = src[start_idx:end_idx]
    p2_off = start_idx
    p2_mod = ("Researchers have identified paraphrase plagiarism in surveys spanning over two decades "
              "(Alzahrani et al. 2012; Foltynek et al. 2019; Maurer et al. 2006; Weber-Wulff 2014).")

    b = DocBuilder()
    b.add(
        "An Overview of Automated Plagiarism Detection Techniques\n\n"
        "Introduction\n\n"
        "Universities and publishers alike rely heavily on automated software to flag potential cases of "
        "academic dishonesty before a human reviewer ever looks at a submission. This essay gives a short "
        "overview of why such automated detection is necessary at scale, and how paraphrase-style plagiarism "
        "in particular has been studied in the research literature.\n\n"
        "Why Automation Is Necessary\n\n"
    )
    b.add_plagiarised(p1_mod, src_id, p1_off, "near-verbatim", "synonym-substitution", source_length=len(p1))
    b.add(
        " No university could realistically employ enough staff to manually compare every submitted "
        "assignment against every other document a student might plausibly have copied from, which is why "
        "automated similarity checking has become a near-universal part of the submission pipeline at most "
        "institutions.\n\n"
        "Paraphrase Plagiarism in the Literature\n\n"
    )
    b.add_plagiarised(p2_mod, src_id, p2_off, "near-verbatim", "synonym-substitution", source_length=len(p2))
    b.add(
        " This body of work consistently finds that simple word substitution is the most common way "
        "students disguise copied material, ahead of more structurally invasive changes such as sentence "
        "reordering.\n\n"
        "How Automated Systems Are Evaluated\n\n"
        "Evaluating a plagiarism detection system requires a labelled benchmark where the true source of "
        "every plagiarised passage is known in advance, which is precisely what shared-task corpora in this "
        "field aim to provide. A system is typically scored on how many plagiarised passages it correctly "
        "identifies relative to how many false alarms it raises on genuinely original text, and these two "
        "numbers trade off against each other depending on how aggressively the system's internal similarity "
        "threshold is set. A system tuned to catch every possible case of copying will inevitably also flag "
        "some passages that merely share common phrasing or discuss a well-known topic using standard "
        "terminology, and tuning away from this produces the opposite problem of missing genuine but subtly "
        "obfuscated copying.\n\n"
        "Why Word Substitution Remains Common\n\n"
        "From a plagiarist's perspective, synonym substitution is attractive precisely because it requires "
        "very little effort: an automated thesaurus tool or word-processor synonym suggestion feature can "
        "perform much of the rewriting without the user needing to understand the underlying material at "
        "all. This stands in contrast to genuine paraphrasing or idea-level restatement, both of which "
        "require the person doing the copying to actually understand the source material well enough to "
        "express its meaning in different words, which is a higher bar and explains why these heavier forms "
        "of obfuscation are observed less frequently in the literature on student plagiarism specifically.\n\n"
        "Limitations of Lexical Substitution Detection\n\n"
        "Detecting synonym substitution reliably is harder than it might initially seem, because not every "
        "wording difference between two passages indicates copying, and not every shared word indicates the "
        "absence of obfuscation. Two independently written passages discussing the same well-known concept "
        "will naturally share a great deal of vocabulary, while a paraphrased copy may, by chance, retain "
        "very little surface overlap with its source if the substitutions are extensive enough. This is part "
        "of why purely lexical detection methods are increasingly supplemented with semantic similarity "
        "measures that compare meaning rather than exact wording, even though semantic methods introduce "
        "their own complications around computational cost and threshold calibration.\n\n"
        "Structural Obfuscation Beyond Word Substitution\n\n"
        "Beyond simple synonym replacement, a more determined plagiarist may restructure a passage at the "
        "sentence or paragraph level, splitting a long sentence into two shorter ones, merging adjacent "
        "sentences, or reordering clauses while preserving the same overall propositional content. These "
        "structural changes survive lexical detection somewhat better than pure synonym substitution does, "
        "since a meaningful fraction of the original vocabulary is typically retained even after the "
        "restructuring, but they defeat detection methods that rely heavily on sentence-level matching units, "
        "since the boundaries of what counts as a matching unit no longer align between the suspicious "
        "passage and its source. Detection systems built to catch this category of obfuscation generally "
        "operate at a coarser granularity, comparing overlapping windows of several sentences rather than "
        "single sentences, specifically so that a sentence boundary shifted by restructuring does not cause "
        "the matching unit itself to fail to align with its counterpart in the source document.\n\n"
        "Translation-Based Obfuscation\n\n"
        "An especially difficult category of obfuscation involves translating a source passage into another "
        "language and back, or simply translating directly into the target document's language from a "
        "source written in a different one. Round-trip translation through an intermediate language "
        "frequently introduces enough lexical and syntactic drift to defeat straightforward lexical matching "
        "entirely, while often preserving the underlying meaning well enough that a human reader would still "
        "recognise the passage as derived from its source. Detecting this category reliably generally "
        "requires either a translation step before comparison or a multilingual semantic representation "
        "capable of recognising paraphrase-level similarity across languages, both of which are considerably "
        "more computationally expensive than monolingual lexical matching and both of which inherit whatever "
        "blind spots the underlying translation or multilingual model carries, particularly for less "
        "well-resourced language pairs.\n\n"
        "Benchmark Corpora for Detection Research\n\n"
        "Much of what is known empirically about how well different detection methods perform on different "
        "obfuscation categories comes from shared-task benchmark corpora that deliberately construct "
        "documents containing passages obfuscated according to a controlled set of strategies, ranging from "
        "no obfuscation at all through light synonym substitution to heavy paraphrasing. These benchmarks "
        "allow systematic comparison of detection methods under matched conditions, but they share a known "
        "limitation: because the obfuscation strategies are constructed according to an explicit rule rather "
        "than arising organically from how real students or authors actually disguise copied material, a "
        "method that performs unusually well on a specific benchmark category may be picking up on artefacts "
        "of how that category was constructed rather than on properties that generalise to real-world "
        "obfuscated plagiarism encountered outside the benchmark.\n\n"
        "Combining Detection Categories in a Single Pipeline\n\n"
        "In practice, a production-grade detection pipeline rarely relies on a single technique tuned for a "
        "single obfuscation category, since real plagiarism in the wild is rarely confined to exactly one "
        "category either; a single plagiarised passage might combine synonym substitution with sentence "
        "reordering and a degree of paraphrasing all at once. Effective pipelines therefore typically run "
        "several detection methods in parallel, each targeting a different obfuscation category, and merge "
        "their outputs through a final scoring or classification stage that decides whether the combined "
        "evidence across methods is strong enough to flag the passage, rather than relying on any single "
        "method's score in isolation. This layered design mirrors the broader cheap-filter-then-expensive-"
        "verifier pattern that recurs throughout the field whenever a single detection strategy proves "
        "insufficient against the full range of obfuscation a real plagiarist might employ.\n\n"
        "Fingerprinting and Approximate Matching at Scale\n\n"
        "When a detection system must compare a suspicious document against a reference corpus numbering in "
        "the millions of documents, exhaustive pairwise comparison becomes computationally infeasible "
        "regardless of how accurate the underlying comparison method is for any single pair. Fingerprinting "
        "techniques address this by selecting a small, carefully chosen subset of hashes from each document, "
        "rather than every possible n-gram hash, using selection strategies designed so that two documents "
        "sharing a substantial overlapping passage are likely to share at least one selected fingerprint even "
        "though the vast majority of their hashes were discarded. This selective hashing makes it practical "
        "to build an inverted index mapping fingerprints back to the documents that contain them, turning the "
        "retrieval problem into a fast lookup rather than an exhaustive scan, at the cost of a small but "
        "non-zero chance of missing a genuine match whose distinguishing fingerprints happened not to survive "
        "the selection process.\n\n"
        "The Specific Case of Code and Mathematical Notation\n\n"
        "Certain categories of academic content resist standard obfuscation-detection techniques for reasons "
        "specific to their notation rather than to the general problem of paraphrase detection. Mathematical "
        "derivations and formulae can be reproduced with identical underlying meaning while varying surface "
        "notation substantially, for instance by renaming variables or reordering algebraically equivalent "
        "terms, which defeats lexical matching in much the same way identifier renaming defeats naive source "
        "code comparison. Detecting plagiarised mathematical content reliably therefore typically requires "
        "normalising expressions into a canonical symbolic form before comparison, analogous to how code "
        "plagiarism detectors normalise source code into an abstract syntax tree stripped of cosmetic "
        "variation, though the normalisation step for mathematics is considerably less standardised across "
        "tools than the equivalent step for common programming languages.\n\n"
        "Institutional Deployment Considerations\n\n"
        "Beyond the purely technical question of which detection method performs best in isolation, "
        "institutions adopting these systems face practical deployment decisions that shape how effective the "
        "chosen method turns out to be once integrated into a real submission workflow. Running detection at "
        "submission time, before a deadline, allows students to see a similarity report and revise their work "
        "accordingly, which can serve an educative function but also gives a sophisticated plagiarist direct "
        "feedback on which obfuscation strategies successfully evade the specific tool in use, effectively "
        "letting them iterate against the detector before final submission. Running detection only after the "
        "deadline avoids handing this feedback to potential plagiarists but removes the educative benefit for "
        "students who would have revised honestly flagged passages in good faith had they known about them "
        "earlier, illustrating that even the timing of when a technically identical detection method is run "
        "carries meaningful, non-technical trade-offs for an institution to weigh.\n\n"
        "False Positive Management at Institutional Scale\n\n"
        "Even a well-tuned detection system deployed across an entire institution will generate a non-trivial "
        "absolute number of false positives simply because of the volume of submissions processed each term, "
        "and how an institution handles this volume of flagged-but-likely-innocent cases matters as much for "
        "perceived fairness as the underlying detection accuracy itself. Institutions that route every flagged "
        "submission through an identical, fairly heavyweight formal review process regardless of how weak the "
        "underlying evidence is risk overwhelming the staff responsible for review and risk subjecting "
        "students with entirely innocent explanations, such as quoting a standard formula or a common turn of "
        "phrase, to a stressful and time-consuming process disproportionate to the strength of the evidence "
        "against them. A tiered review process, in which the strength of the automated similarity evidence "
        "determines how much scrutiny and how formal a process a given case receives, can reduce this burden "
        "considerably, though designing the tiers themselves requires the same kind of careful threshold "
        "calibration already discussed for the underlying detection signal.\n\n"
        "Detection Tool Vendor Diversity and Cross-Validation\n\n"
        "A practical observation from institutions that have compared multiple commercial detection products "
        "side by side is that different vendors' tools, even when nominally solving the same problem, "
        "disagree noticeably on which passages they flag and at what similarity score, reflecting differences "
        "in their underlying reference corpora, their specific n-gram or embedding configuration, and "
        "undisclosed proprietary tuning choices. This disagreement is not necessarily a sign that one tool is "
        "simply better than another; it more often reflects genuine differences in what each tool's reference "
        "corpus actually contains, since a tool cannot flag a match against a source document it does not "
        "have indexed in the first place. Some institutions have responded by running more than one detection "
        "tool in parallel and treating agreement between independent tools as a stronger signal than either "
        "tool's output alone, accepting the added licensing and processing cost in exchange for a modest "
        "improvement in confidence about the cases that genuinely warrant escalation to a human reviewer.\n\n"
        "Detecting Self-Plagiarism and Recycled Submissions\n\n"
        "A category of misconduct that sits somewhat apart from the obfuscation taxonomy discussed so far is "
        "self-plagiarism, where a student submits work substantially identical to something they themselves "
        "previously submitted, either for a different course or an earlier attempt at the same assignment, "
        "without disclosing the reuse. Detecting this category requires a reference corpus that includes a "
        "given student's own prior submissions across the institution, which raises its own data governance "
        "questions about how long student work should be retained for this purpose and who should have access "
        "to a cross-course submission history. From a purely technical detection standpoint, self-plagiarism "
        "is often easier to catch than externally sourced plagiarism, since the matching passage is by "
        "definition verbatim or very close to verbatim against a known, fully indexed source, but it raises "
        "harder policy questions about whether reusing one's own prior work should be treated with the same "
        "severity as reusing someone else's, given that the underlying concern, namely whether a student is "
        "being assessed on genuinely new work, differs somewhat between the two cases even though both "
        "involve presenting previously existing material as newly produced.\n\n"
        "Conclusion\n\n"
        "Automated detection systems exist because manual review does not scale, and the research literature "
        "on paraphrase plagiarism gives these systems a useful map of which kinds of obfuscation are most "
        "common and therefore most worth detecting reliably, even as the rise of more sophisticated rewriting "
        "tools continues to push the field toward semantic rather than purely lexical detection strategies.\n"
    )
    write_doc("suspicious-document00008", b)


def build_doc_00014() -> None:
    """Near-verbatim — sentence reordering within paragraph."""
    src_id = "source-document00018"
    src = source_text(src_id)
    p1 = ("The problem of community detection requires the partition of a network into communities of "
          "densely connected nodes, with the nodes belonging to different communities being only sparsely "
          "connected. Precise formulations of this optimization problem are known to be computationally "
          "intractable. Several algorithms have therefore been proposed to find reasonably good partitions "
          "in a reasonably fast way.")
    p1_off = find_offset(src, p1, src_id)
    p1_reordered = (
        "Several algorithms have therefore been proposed to find reasonably good partitions in a reasonably "
        "fast way. Precise formulations of this optimization problem are known to be computationally "
        "intractable. The problem of community detection requires the partition of a network into "
        "communities of densely connected nodes, with the nodes belonging to different communities being "
        "only sparsely connected."
    )

    p2 = ("Modularity has been used to compare the quality of the partitions obtained by different methods, "
          "but also as an objective function to optimize [13]. Unfortunately, exact modularity optimization "
          "is a problem that is computationally hard [14] and so approximation algorithms are necessary when "
          "dealing with large networks.")
    p2_off = find_offset(src, p2, src_id)
    p2_reordered = (
        "Exact modularity optimization is, unfortunately, a problem that is computationally hard, which means "
        "approximation algorithms are necessary when dealing with large networks. Modularity has nonetheless "
        "been widely used to compare the quality of partitions obtained by different methods, and also as an "
        "objective function to optimize in its own right."
    )

    b = DocBuilder()
    b.add(
        "Modularity Optimisation: Why Exact Solutions Do Not Scale\n\n"
        "Introduction\n\n"
        "Community detection algorithms are judged largely by how well they balance solution quality against "
        "computational cost on networks with millions of nodes. This essay focuses specifically on why exact "
        "optimisation of the modularity objective is infeasible in practice and what that implies for "
        "algorithm design.\n\n"
        "The Partitioning Problem\n\n"
    )
    b.add_plagiarised(p1_reordered, src_id, p1_off, "near-verbatim", "sentence-reordering", source_length=len(p1))
    b.add(
        " This intractability is not a minor technical inconvenience; it is the central reason why every "
        "practical community detection algorithm in wide use today is a heuristic rather than an exact "
        "solver.\n\n"
        "Modularity as an Objective\n\n"
    )
    b.add_plagiarised(p2_reordered, src_id, p2_off, "near-verbatim", "sentence-reordering", source_length=len(p2))
    b.add(
        " Approximation therefore is not a fallback option chosen for convenience, but the only realistic way "
        "to apply modularity-based community detection to networks of any meaningful size.\n\n"
        "Why Greedy Local Moves Work Well in Practice\n\n"
        "It is worth dwelling on why a greedy, purely local heuristic — repeatedly moving a single node into "
        "whichever neighbouring community most increases modularity — performs so well despite offering no "
        "global optimality guarantee. Empirically, real-world networks tend to have a strong underlying "
        "community structure that is locally detectable: a node's correct community membership is usually "
        "evident just from looking at its immediate neighbours, without needing to reason about the network "
        "as a whole. This locality is precisely what greedy local-move heuristics exploit, and it explains "
        "why such heuristics tend to find partitions close to the best achievable modularity even though "
        "they never explicitly search the full space of possible partitions.\n\n"
        "The Cost of the Hierarchical Aggregation Step\n\n"
        "A second source of the method's efficiency is the aggregation phase, where communities found in one "
        "pass become the nodes of a new, much smaller network for the next pass. Because the number of "
        "distinct communities is typically far smaller than the number of original nodes, this aggregated "
        "network shrinks rapidly across passes, meaning that the computationally expensive first pass — run "
        "on the full original network — dominates total running time, while subsequent passes are cheap by "
        "comparison. This is part of why the method scales to networks with hundreds of millions of edges: "
        "most of the work happens once, on a single pass over the raw input, rather than being repeated at "
        "full scale multiple times.\n\n"
        "Sensitivity to Node Ordering\n\n"
        "One known property of greedy local-move methods is that the order in which nodes are processed can "
        "affect the specific partition found, even though it tends not to affect the resulting modularity "
        "score by much in practice. This sensitivity to ordering means that two runs of the same algorithm on "
        "the same network, differing only in the order nodes were visited, can produce partitions that "
        "disagree on community boundaries for a small fraction of nodes near those boundaries, even when "
        "both partitions achieve very similar overall modularity. For applications where the exact community "
        "assignment of borderline nodes matters, this non-determinism is worth taking into account when "
        "interpreting results.\n\n"
        "The Resolution Limit of Modularity\n\n"
        "Modularity optimisation, despite its widespread use, has a known structural weakness called the "
        "resolution limit, which causes the metric to systematically fail to detect small, well-defined "
        "communities once they are embedded within a sufficiently large network, even when those communities "
        "are intuitively obvious to a human examining the relevant subgraph in isolation. The underlying "
        "cause is that modularity compares the observed density of edges within a candidate community against "
        "an expected density under a null model defined relative to the whole network, and as the whole "
        "network grows, the expected density baseline shifts in a way that makes merging two genuinely "
        "distinct small communities into one larger community appear, by the modularity score alone, to be an "
        "improvement, even though the merge destroys real structural information. Several proposed extensions "
        "address this limitation by introducing an explicit resolution parameter that lets a user control the "
        "characteristic scale of detected communities, trading away modularity's original appeal as a "
        "parameter-free metric in exchange for the ability to recover communities the unmodified metric "
        "would systematically miss.\n\n"
        "Comparing Modularity-Based Methods Against Alternative Objectives\n\n"
        "Modularity is not the only objective function used for community detection, and understanding its "
        "specific trade-offs requires situating it against alternatives such as stochastic block models, "
        "which take an explicitly probabilistic approach and model the network as having been generated by an "
        "underlying community structure with given inter- and intra-community connection probabilities. "
        "Stochastic block model approaches can in principle avoid the resolution limit and provide a "
        "statistically principled way to determine the number of communities present, but fitting such a "
        "model is itself a harder computational problem than greedy modularity optimisation, generally "
        "requiring iterative inference procedures that scale considerably worse to networks with hundreds of "
        "millions of edges. This trade-off between statistical principledness and computational tractability "
        "recurs throughout the community detection literature: methods that more faithfully model the "
        "generative process underlying a real network tend to be the same methods that scale worst to the "
        "network sizes practitioners actually need to analyse.\n\n"
        "Empirical Benchmarking Against Synthetic Networks\n\n"
        "Because real networks rarely come with a verified ground-truth community structure against which a "
        "candidate partition can be checked, much of the empirical evaluation of community detection "
        "algorithms relies on synthetic benchmark networks generated according to a known planted community "
        "structure, allowing a researcher to measure directly how well a given algorithm recovers the "
        "communities that were used to generate the network in the first place. These synthetic benchmarks "
        "are useful precisely because they provide unambiguous ground truth, but they share a limitation "
        "familiar from other areas of this field: a synthetic network generated according to a particular "
        "statistical model may not capture every structural property of real-world networks, such as "
        "heavy-tailed degree distributions or hierarchical nesting of communities within communities, and an "
        "algorithm that performs well specifically on the synthetic benchmark's particular generative "
        "assumptions may not generalise as well to real networks whose structure departs from those "
        "assumptions in ways the benchmark does not capture.\n\n"
        "Parallel and Distributed Implementations\n\n"
        "As networks have grown to the scale of major social platforms, single-machine implementations of "
        "even efficient heuristic algorithms eventually hit a hard memory or runtime ceiling, motivating "
        "parallel and distributed implementations that partition the network across multiple machines. "
        "Distributing a greedy local-move algorithm is not entirely straightforward, since the algorithm's "
        "core operation, moving a single node into whichever neighbouring community most increases global "
        "modularity, requires knowledge of the current community assignments of that node's neighbours, which "
        "may reside on a different machine if the network has been partitioned naively across the cluster. "
        "Practical distributed implementations therefore typically batch many node-move decisions together "
        "and synchronise periodically across machines, accepting a degree of staleness in the community "
        "assignments used to make any individual decision in exchange for avoiding the prohibitive "
        "communication cost of synchronising after every single node move.\n\n"
        "Stability of Detected Communities Across Algorithm Choice\n\n"
        "A practical question facing anyone applying community detection to a real dataset is how much the "
        "resulting partition depends on which specific algorithm was used, given that several different "
        "heuristics all claim to approximately optimise modularity or a related objective. Empirical "
        "comparisons across algorithms on the same real-world networks generally find substantial agreement "
        "on the coarse-grained community structure, particularly for large, well-separated communities, but "
        "meaningfully more disagreement at the boundaries between communities and for smaller or more weakly "
        "separated groups, echoing the node-ordering sensitivity already discussed within a single algorithm. "
        "This pattern suggests that the broad community structure recovered by modularity-based methods is a "
        "fairly robust property of the underlying network, while the precise treatment of borderline nodes "
        "should be treated with appropriate caution regardless of which specific algorithm produced the "
        "reported partition.\n\n"
        "Hierarchical Structure and Multi-Resolution Analysis\n\n"
        "Real networks frequently exhibit community structure at more than one scale simultaneously, with "
        "small, tightly bound groups nesting inside progressively larger and looser groupings, a pattern most "
        "visible in social networks where small friend groups nest inside broader social circles which "
        "themselves nest inside still broader affiliations such as a shared workplace or institution. A single "
        "flat partition produced by standard modularity optimisation necessarily picks one particular scale "
        "to report, discarding information about structure at other scales that may be equally meaningful "
        "depending on the analysis question being asked. The two-phase aggregation procedure that makes "
        "greedy modularity optimisation efficient happens to produce a natural byproduct relevant to this "
        "problem: the sequence of partitions found at each pass before final convergence forms an implicit "
        "hierarchy, with early passes capturing fine-grained local structure and later passes capturing "
        "progressively coarser structure, giving an analyst a multi-resolution view of the network's "
        "community structure essentially for free as a side effect of the optimisation procedure rather than "
        "as a separately engineered feature.\n\n"
        "Validating Detected Communities Against External Metadata\n\n"
        "Where a network comes with metadata describing its nodes independent of the network's edge "
        "structure, such as known organisational affiliation, geographic location or topic labels, this "
        "metadata offers an external validation route distinct from purely structural metrics like "
        "modularity. A detected community that aligns closely with an independently known metadata attribute, "
        "such as nearly all of a community's members sharing the same organisational department, provides "
        "considerably more confidence that the detected grouping reflects a real, substantively meaningful "
        "structure rather than an artefact of the optimisation procedure. The absence of such alignment does "
        "not necessarily indicate a poor partition, since genuine community structure in a network need not "
        "correspond to any single recorded metadata attribute, but where alignment is available and strong, "
        "it provides a valuable independent check that purely internal structural metrics cannot offer on "
        "their own.\n\n"
        "Implications for Algorithm Selection in Applied Settings\n\n"
        "For a practitioner facing a concrete applied problem rather than a methodological research question, "
        "the accumulated body of work on modularity optimisation's limitations translates into a fairly "
        "practical set of recommendations. Where the network of interest is known to contain communities "
        "spanning a wide range of sizes, a resolution-aware variant should generally be preferred over the "
        "unmodified objective, since the resolution limit's failure mode, namely silently merging small "
        "genuine communities into larger ones, is difficult to detect after the fact without already knowing "
        "what the correct answer should have been. Where computational budget is tightly constrained and the "
        "network is enormous, the standard greedy two-phase heuristic remains the most practical default "
        "despite its known sensitivity to node ordering, simply because the available alternatives that "
        "better address that sensitivity tend to cost substantially more in runtime than the modest gain in "
        "partition stability is usually worth for a one-off exploratory analysis. These are, admittedly, "
        "rules of thumb rather than principled guarantees, but the absence of a single dominant method across "
        "every use case is itself the central finding of this entire line of research, and treating any one "
        "algorithm as a universal default ignores exactly the trade-offs this essay has tried to make "
        "explicit.\n\n"
        "Software Ecosystem Support and Practical Accessibility\n\n"
        "A final, less theoretical consideration shaping which method actually gets used in applied work is "
        "the maturity of available software implementations. Widely used network analysis libraries typically "
        "ship a well-tested implementation of the standard greedy modularity heuristic as a default option, "
        "making it the path of least resistance for a practitioner without specialised expertise in the "
        "underlying algorithmic literature, while resolution-aware variants and stochastic block model "
        "inference are less uniformly available and sometimes require separate, less actively maintained "
        "packages. This accessibility gap means that, in practice, the choice of community detection method "
        "for a given applied project is often driven as much by what is conveniently available in a "
        "practitioner's existing toolchain as by a deliberate, fully informed weighing of the trade-offs "
        "described throughout this essay, which is worth bearing in mind when interpreting why the standard "
        "heuristic remains so dominant in applied work despite its documented limitations.\n\n"
        "Conclusion\n\n"
        "Accepting that exact modularity optimisation is computationally hard reframes the design question "
        "for community detection algorithms: the goal is not to find the provably best partition, but to "
        "find a good enough partition within a time budget that scales acceptably with network size, and the "
        "empirical success of greedy local-move heuristics suggests that this reframing costs very little in "
        "practice on networks with genuine underlying community structure.\n"
    )
    write_doc("suspicious-document00009", b)


def build_doc_00015() -> None:
    """Paraphrase + idea-level + mixed, multi-source — heavier obfuscation document."""
    src1_id = "source-document00019"
    src1 = source_text(src1_id)
    p1 = ("Assessment is an integral part of higher education, serving as a means of evaluating "
          "student learning and progress. There are many different forms of assessment, including "
          "exams, papers, projects, and presentations, and they can be used to assess a wide range of "
          "learning outcomes, such as knowledge, skills, and attitudes.")
    p1_off = find_offset(src1, p1, src1_id)
    p1_paraphrase = (
        "Evaluation sits at the core of higher education, functioning as the primary mechanism by which "
        "institutions gauge how much a student has actually learned. Universities draw on a wide variety of "
        "evaluation formats — written exams, term papers, group projects, oral presentations — each suited "
        "to measuring a different combination of factual knowledge, practical skill, and professional "
        "attitude."
    )

    src29_id = "source-document00029"
    src29 = source_text(src29_id)
    marker2 = "out of 15 available plagiarism detection tools, none satisfied their criteria of being labelled"
    idx2 = src29.find(marker2)
    if idx2 == -1:
        raise ValueError("marker2 not found in source-document00029")
    line_start = src29.rfind("\n", 0, idx2) + 1
    sentence_end = src29.find(".", idx2) + 1
    p2 = src29[line_start:sentence_end].strip()
    p2_off = src29.find(p2, line_start)
    p2_src_id = src29_id
    p2_paraphrase = (
        "An evaluation study examining commercially available plagiarism checkers concluded that not a "
        "single one of the fifteen tools tested met the bar the authors had set for what counts as genuinely "
        "useful software."
    )

    src2_id = "source-document00016"
    src2 = source_text(src2_id)
    p3 = ("As  image  databases  usually  contain  unlabelled  images, "
          "these can be exploited to help supervised learning by asking the  user  to  label  them.  The  "
          "goal  is  naturally  to  minimize asking  help  from  the  user.")
    p3_off = find_offset(src2, p3, src2_id)
    p3_idea = (
        "Most real-world image collections used for retrieval contain far more unlabelled examples than "
        "labelled ones, and systems that ask a human to manually label every image quickly run into a "
        "scalability wall. A more sustainable design goal is therefore to extract as much value as possible "
        "from the unlabelled portion of the collection, reserving requests for human input for only the "
        "cases where the system is genuinely uncertain."
    )

    b = DocBuilder()
    b.add(
        "Assessment Design, Detection Reliability, and Scalable Human Feedback: Three Connected Problems\n\n"
        "Introduction\n\n"
        "This essay draws together three topics that initially appear to belong to separate subfields — "
        "assessment design in higher education, the reliability of plagiarism detection software, and "
        "scalable use of human feedback in machine learning systems — and argues that they share a common "
        "underlying tension between the cost of human judgment and the desire to automate decisions at "
        "scale.\n\n"
        "Assessment as a Multi-Format Problem\n\n"
    )
    b.add_plagiarised(p1_paraphrase, src1_id, p1_off, "paraphrase", "heavy-rewrite", source_length=len(p1))
    b.add(
        " The diversity of formats exists precisely because no single assessment type captures every kind of "
        "learning outcome an institution cares about, which is itself an early example of the broader "
        "pattern this essay traces.\n\n"
        "Detection Software Does Not Close the Loop Alone\n\n"
    )
    b.add_plagiarised(p2_paraphrase, p2_src_id, p2_off, "paraphrase", "heavy-rewrite", source_length=len(p2))
    b.add(
        " If automated tools cannot be trusted to make a final determination on their own, institutions are "
        "forced back toward exactly the kind of expensive human review that automation was supposed to "
        "reduce, just applied selectively to the cases the software flags rather than to every submission.\n\n"
        "Scalable Human Feedback in Retrieval Systems\n\n"
    )
    b.add_plagiarised(p3_idea, src2_id, p3_off, "idea-level", "concept-reuse", source_length=len(p3))
    b.add(
        "\n\nThe same logic that governs how an image retrieval system should spend a limited budget of human "
        "labelling effort also governs how an academic institution should spend a limited budget of human "
        "review time when assessing flagged submissions: in both cases, the system should be designed to "
        "request human input only where automated confidence is genuinely low, rather than treating every "
        "case identically.\n\n"
        "A Common Architecture: Cheap Filter, Expensive Verifier\n\n"
        "Looking across all three domains, a recurring architectural pattern emerges. A cheap, high-recall "
        "first stage processes every case and produces a confidence score: a similarity score for a "
        "submitted assignment, an uncertainty score for an unlabelled image, or a rubric-based estimate for "
        "an exam answer. Cases that receive a confident automated decision in either direction — clearly "
        "fine, or clearly problematic — are resolved without further human involvement, while only the "
        "ambiguous middle band is escalated to a human reviewer, an expert labeller, or a senior grader. This "
        "architecture is attractive precisely because it concentrates the scarcest resource, human attention, "
        "on exactly the cases where it adds the most value, while letting cheap automation handle the high "
        "volume of unambiguous cases.\n\n"
        "Where the Pattern Breaks Down\n\n"
        "The cheap-filter-then-expensive-verifier pattern works best when the automated confidence score is "
        "well-calibrated, meaning a high confidence score genuinely corresponds to a high probability of "
        "being correct. When calibration is poor — for instance, when a plagiarism detector is confidently "
        "wrong about a passage that merely shares common technical vocabulary with its source, or when an "
        "image classifier is confidently wrong about a visually ambiguous photograph — the architecture can "
        "systematically under-escalate exactly the cases that most need human review. This is why systems "
        "built on this pattern usually require ongoing recalibration against ground truth labels collected "
        "after the fact, rather than being calibrated once at deployment and left unchanged.\n\n"
        "Institutional Costs of Getting the Boundary Wrong\n\n"
        "Setting the escalation boundary too narrow, so that almost every case is escalated to a human, "
        "defeats the purpose of automation and simply reproduces the original scalability problem the system "
        "was built to solve. Setting it too wide, so that almost nothing is escalated, risks systematic "
        "errors going unnoticed for long periods, since the automated component is, by construction, never "
        "checked against human judgment in the cases it handles confidently. Institutions adopting any of "
        "these systems therefore need an explicit, periodically revisited policy on where this boundary sits, "
        "rather than treating the automated component's default threshold as a fixed, neutral setting.\n\n"
        "Feedback Loops Between Assessment Design and Detection Capability\n\n"
        "A subtler interaction connects assessment design directly back to detection reliability, beyond the "
        "simple observation that some assessment formats are easier for a generative model to complete than "
        "others. When an institution redesigns an assessment specifically to be harder for automated tools to "
        "complete convincingly, it simultaneously narrows the space of plausible legitimate answers, which "
        "can paradoxically make automated similarity-based detection more reliable for whatever submissions do "
        "still resemble each other suspiciously closely, since genuine topical overlap becomes rarer once an "
        "assessment is built around an unusually specific or recent prompt. Conversely, a generic assessment "
        "prompt reused across many years and many institutions provides a much richer pool of prior "
        "submissions and online material for a model or a student to draw on, which simultaneously makes "
        "automated tools more effective at generating plausible answers and makes detection of any specific "
        "instance of reuse harder, since legitimate independent overlap with that vast prior pool of similar "
        "submissions is also more likely. Assessment design choices therefore affect detection reliability "
        "indirectly, by reshaping the statistical baseline against which any specific submission's similarity "
        "score must be judged.\n\n"
        "The Cost of False Confidence in Any Single Layer\n\n"
        "A theme implicit throughout the comparison drawn in this essay deserves to be made explicit: each of "
        "the three domains discussed contains at least one stage where an institution or a system designer "
        "might be tempted to treat an automated signal as more authoritative than it actually is, simply "
        "because the signal is presented as a precise number. A similarity score, an AI-detection confidence "
        "value, and an automated image-relevance estimate all share the property that their numeric precision "
        "can outstrip their actual reliability, particularly near the decision boundary where the consequences "
        "of an error are highest. Designing assessment, detection and retrieval systems responsibly therefore "
        "requires building in deliberate scepticism toward exactly the signals the system relies on most "
        "heavily, rather than treating a high-precision-looking number as self-validating simply because it "
        "was produced by a sophisticated underlying model.\n\n"
        "Toward Shared Infrastructure Across the Three Domains\n\n"
        "Given how much structural similarity exists across these three ostensibly separate domains, there is "
        "a reasonable case for treating the underlying cheap-filter-then-expensive-verifier architecture as "
        "shared infrastructure rather than reinventing it independently within each application. A generic "
        "framework for routing cases between automated and human decision-makers based on a calibrated "
        "confidence score, with explicit support for periodic recalibration against newly collected ground "
        "truth, could in principle serve an academic integrity office, an image retrieval team, and an "
        "assessment design committee with only modest domain-specific customisation at the edges. Whether "
        "such shared infrastructure actually emerges in practice likely depends less on the underlying "
        "technical feasibility, which appears reasonably strong given the structural parallels already "
        "discussed, than on whether institutions and vendors operating in these otherwise disconnected fields "
        "ever have sufficient reason to recognise the parallel and act on it together.\n\n"
        "Cold-Start Problems Across All Three Domains\n\n"
        "Each of the three domains discussed in this essay also shares a common cold-start difficulty: a "
        "system has no calibrated confidence score to rely on until it has accumulated enough labelled "
        "examples to learn what a reliable signal actually looks like in that specific deployment context. A "
        "newly deployed plagiarism detector at an institution with no prior flagged-and-reviewed case history "
        "has no local data from which to calibrate a confidence threshold and must initially borrow "
        "calibration from some other institution's data, which may not transfer cleanly given the domain "
        "dependence of similarity thresholds discussed elsewhere. A newly launched image retrieval system "
        "faces an analogous problem in deciding which images are worth prioritising for human labelling "
        "before any labels exist at all, typically resorting to some form of unsupervised pre-clustering to "
        "make an initial, unprincipled guess about where to direct early labelling effort. An institution "
        "designing an assessment for a brand-new course format faces the same basic difficulty in calibrating "
        "what counts as a suspiciously generic, possibly AI-generated submission, since no prior cohort's "
        "submissions exist yet to establish a local baseline. In all three cases, the practical workaround is "
        "broadly similar: operate conservatively during an initial bootstrapping period, deliberately "
        "over-escalating to human judgment until enough confirmed cases accumulate to support a properly "
        "calibrated automated threshold.\n\n"
        "Auditing Automated Decisions After the Fact\n\n"
        "A final shared concern across all three domains is the need for retrospective auditing of automated "
        "decisions, independent of whatever confidence calibration was used at decision time. Even a "
        "well-calibrated system will make some number of confident errors simply because calibration is a "
        "statement about aggregate behaviour across many cases rather than a guarantee about any individual "
        "case, and an institution that never revisits past automated decisions has no way of discovering "
        "whether its system's actual error rate matches its theoretical calibration or has drifted away from "
        "it due to changes in the underlying population. Periodic retrospective audits, in which a sample of "
        "past automated decisions is re-examined by a human reviewer blind to the system's original output, "
        "provide the only reliable mechanism for catching this kind of silent drift before it accumulates "
        "into a serious and possibly systematically biased pattern of errors.\n\n"
        "Transferability of Calibration Across Institutions and Deployments\n\n"
        "A recurring practical question across all three domains is whether a calibration established at one "
        "institution, on one dataset, or for one deployment can be reused elsewhere without substantial "
        "re-validation. The answer in each case tends to be a qualified no: a confidence threshold tuned "
        "against one university's historical case mix reflects that institution's specific student "
        "population, assessment style and disciplinary mix, and transplanting it unchanged to a different "
        "institution risks the same domain-mismatch problems already discussed in the context of similarity "
        "thresholds more generally. The same caution applies to an image retrieval system's confidence "
        "calibration when the system is deployed against a new image collection with different visual "
        "characteristics than the one used for calibration, and to an assessment rubric's automated "
        "confidence scoring when applied to a new course with a different student population than the one the "
        "rubric was originally validated against. This persistent need for local re-validation is one of the "
        "more underappreciated costs of deploying any of these systems at scale across multiple sites, since "
        "it is tempting, and often administratively convenient, to treat a single validated configuration as "
        "portable when the underlying statistical assumptions rarely transfer cleanly.\n\n"
        "Documentation and Explainability as Shared Requirements\n\n"
        "A final point worth raising is that all three domains increasingly face external pressure, whether "
        "regulatory, institutional or simply reputational, to explain automated decisions in terms a "
        "non-specialist can follow, rather than presenting a bare confidence score without context. A student "
        "challenging a plagiarism finding, a curator questioning why a particular image was prioritised for "
        "human labelling, and a grading committee reviewing why an automated rubric scored a borderline "
        "submission a certain way all benefit from systems designed from the outset to produce an "
        "interpretable account of their reasoning, not merely a numeric output. Retrofitting explainability "
        "onto a system designed purely to optimise a confidence score is considerably harder than designing "
        "for it from the start, which is a strong practical argument for treating explainability as a core "
        "design requirement across all three domains rather than as an afterthought addressed only once "
        "external pressure makes it unavoidable.\n\n"
        "Workforce and Training Implications\n\n"
        "Operating any of these systems responsibly over the long term requires staff who understand both the "
        "statistical behaviour of the underlying automated component and the substantive domain in which it "
        "is deployed, a combination of skills that is not yet a standard part of training for academic "
        "integrity officers, library staff, or curatorial teams in image collections. An integrity officer who "
        "understands disciplinary policy thoroughly but treats a vendor's similarity score as a black box "
        "number to be trusted at face value is poorly positioned to recognise when that score is behaving "
        "unreliably for a particular case, while a data scientist who understands the statistical properties "
        "of a confidence score thoroughly but lacks grounding in the substantive domain may calibrate a "
        "threshold that is statistically sound but practically tone-deaf to what actually matters in that "
        "domain's specific decisions. Building institutional capacity that bridges this gap, whether through "
        "dedicated training, cross-functional review committees, or hybrid roles explicitly designed to "
        "combine both kinds of expertise, is likely to matter more for the long-run reliability of these "
        "systems than any single improvement to the underlying algorithms themselves.\n\n"
        "Conclusion\n\n"
        "Whether the domain is grading, integrity checking, or image labelling, the recurring design "
        "challenge is the same: human judgment is accurate but expensive, automated judgment is cheap but "
        "imperfect, and the best systems are the ones that route cases between the two based on how confident "
        "the automated component actually is, rather than relying on either one exclusively, provided that "
        "confidence estimate itself remains well-calibrated over time.\n"
    )
    write_doc("suspicious-document00010", b)


def main() -> None:
    build_clean_docs()
    build_doc_00011()
    build_doc_00012()
    build_doc_00013()
    build_doc_00014()
    build_doc_00015()


if __name__ == "__main__":
    main()
