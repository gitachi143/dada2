# DADA2 — Agentic RAG over SEC filings

**Fill in both sections at the bottom before you submit.** Keep the deployed
app line exactly in this form; it is how your app gets found and graded.

## Getting started

There is no separate spec. The instructions are in the notebooks, and
`DADA2_00_Start_Here.ipynb` is the one to open first: it explains the
assignment, the setup, the grading, and what each of the four notebooks after
it asks for. Work through them in order.

Get the files the same way as every class day: pull, then work in a copy:

```bash
cd lecture-files
git pull
cd ..
cp -r lecture-files/DADA2 DADA2     # work in the copy, not in lecture-files/
cd DADA2
cp example.env .env                  # then fill in your keys
```

Then open `DADA2_00_Start_Here.ipynb` from your `DADA2` copy and run its
setup check.

Deploying is `DEPLOY.md`.

---

## Submission

```
Deployed app: <your Render URL>
```

ADD YOUR NAME TO YOUR RENDER PROJECT SO THE LINK CAN BE PERSONALIZED.

PLEASE ENSURE YOUR URL WORKS BY TESTING IT IN INCOGNITO MODE.

### Write-up

Paste the write-up from notebook 04 (TODO 4.3) here, under the same five
headings.

**Conditions.**
All runs used the DADA2 SEC corpus for NVDA, AMD, and MU with a Pinecone index built from Cloudflare embeddings, 1200-character chunks and 150 overlap, and k = 5 passages passed to the model. Strategies that rerank used Cohere rerank-v3.5. The chat model was gemini-3.5-flash-lite and the decision model was clef-flash. Section 4.1 scored routing on 24 labeled questions (14 specific and 10 comparative). Section 4.2 compared full agents on 14 questions (eight specific and six comparative).

**The routers.**
The decision model is the part of the system that picks a category and says how sure it is using a decision model Clef in our case. The decision model’s decide.scope got 20 of 24 labels right at about 0.50 seconds per call. The chat scoper got 18 of 24 at about 0.79 seconds per call. The decision model was slightly more accurate and faster on this set.

**The agents.**
On the 14-question subset, the ladder agent scored 0.86 judge accuracy with 4.9 chat calls and 16.2 seconds per question, and no decision calls. The diagnosis agent from notebook 03 scored 0.64 with 2.9 chat calls, 2.4 decision calls, and 8.4 seconds per question. The diagnosis agent was cheaper and faster but missed roughly three more questions than the ladder. The ladder keeps escalating search until a chat grader accepts the passages; the diagnosis agent often stops after one diagnose step on sufficient or absent, or after a single stronger step, and sometimes answers before retrieval has covered the full question. The largest clear split was on “How has NVIDIA’s exposure to export controls changed over the last three years?” The ladder was judged correct after escalating retrieval, while the diagnosis agent was judged incorrect because it claimed the passages did not cover NVIDIA’s export controls even though the filings do.

**What did not help.**
On the 14-question agent comparison, the piece that did not pay off was trusting decide.diagnose to decide when search was done. The ladder keeps running retrieval and moving up to filter+rrf+rerank until a chat grader accepts the passages, which cost 4.9 chat calls and 16.2 seconds per question but reached 0.86 accuracy. Notebook 03 agent often stopped after one or two passes when diagnose returned sufficient or absent, or after a single stronger step, which cut cost to 2.9 chat calls and 8.4 seconds but dropped accuracy to 0.64. 

**What I would ship.**
I would ship a split graph: figure → lookup → answer, and everything else → retrieve → answer, using notebook 02 strategies behind retrieve. I would keep decision routing (scope picks the first strategy) but not a single diagnose-and-stop loop as the only control I would require escalation (stronger chain or a short ladder) before answering, especially for comparative questions because comparative questions need passages from more than one company, and one search pass often only covers one side. Our agent table showed the ladder kept escalating and scored 0.86, while diagnose-and-stop often quit early at 0.64, so I would not let comparative items go straight to answer after a single retrieve. And I would want to know whether users care more about fast, cheap answers or maximum correctness, because our tables show the diagnosis agent is quicker and uses fewer chat calls while the ladder is slower but more accurate.