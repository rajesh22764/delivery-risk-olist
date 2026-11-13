# Will my order arrive late? How I built a late-delivery predictor, explained simply

*A learning project on real online-shopping data. I wrote this so that anyone who has finished Class 12 can follow it, even if they have never seen machine learning before.*

---

## 1. The idea in one minute

When you order something online, the website says, "Delivery by 20 August." Sometimes the parcel comes on time. Sometimes it comes late, and you are annoyed.

I asked a simple question: **at the exact moment someone places an order, can a computer guess whether it will arrive after the promised date?**

If the answer is yes, a shop could act early. It could use a faster courier for risky orders, promise a more realistic date, or message the customer before they get angry.

So I built a small "late-delivery predictor" and tested it honestly. This post explains how it works, what I found, and where it fails. The failures turned out to be the most interesting part.

---

## 2. Why this matters

Late deliveries hurt the shop. In my data, customers whose parcels came on time gave an average rating of **4.3 out of 5 stars**. Customers whose parcels came late gave **2.3 stars**. More than half of the late orders got a **1-star review**, compared with only 7% of on-time orders.

So one missed date costs about two stars. That is a lot.

---

## 3. The data I used

I used a public dataset from a Brazilian online marketplace called **Olist**. It has about **100,000 real orders** from 2016 to 2018. For each order it tells me who bought, what they bought, the price, which seller sent it, where the seller and the customer live, when it was ordered, the date promised, and the date it actually arrived.

Two things were missing that I thought might matter, so I **collected them myself** from free websites:

- **Public holidays** in Brazil. A holiday before the delivery date might slow things down.
- **Daily rainfall** for every Brazilian state. Heavy rain might delay trucks.

Think of it like preparing for a project: you start with the main textbook, then add notes from other sources.

This is Brazilian data from several years ago. It teaches how to do the work properly. It does **not** tell us anything about Indian shops.

---

## 4. What does "late" mean? (a small trap)

You would think "late" is obvious: arrived after the promised date. But there was a trap.

The promised date is stored as a date only, like "20 August", which the computer treats as 12:00 at midnight. The arrival is stored with a time, like "20 August, 10:00 in the morning."

If I simply asked "is the arrival time bigger than the promised time?", a parcel that arrived at 10 am **on** the promised day would be counted as late. That is wrong, because the customer got it on the day promised.

So I compare **days**, not times. With the naive method, 8.1% of orders looked late. With the correct method, **6.8%** were late. The difference was just a labelling mistake. I wrote an automatic check so this mistake can never sneak back in.

---

## 5. What the data told me (before building anything)

Before building a model, I explored the data, like studying the question paper before writing an answer. Five things stood out.

**1. Late deliveries come in waves, not at a steady rate.** For most of 2017 about 3% of orders were late. Then it jumped to 12% around Black Friday (a big sale), 14% in February 2018, and **19% in March 2018**. Then it fell back to a few percent. Something happens that affects many orders at once, and then it passes.

**2. The promised dates are padded.** The typical promise was **24 days**, but the typical parcel took only **10 days**. So the shop gives itself a big cushion. Even so, 6.8% still miss the date. Also, very short promises (10 days or less) were late **10.5%** of the time, because there is no spare room, while very long promises (40+ days) were late only **2.4%** of the time.

**3. Distance matters.** If the seller and customer are within 100 km, about **4.5%** of parcels are late. If they are more than 1,500 km apart, about **11.7%** are late.

**4. A few sellers cause most late orders, but they are not always the same sellers.** The worst 10% of sellers were responsible for about **74%** of the late orders. You might think, "Easy! Just find the bad sellers." But when I checked whether the sellers who were late in one period were also late in the next period, the link was **very weak** (a score of about 0.2 on a scale where 1 means "exactly the same sellers", and 0 means "no connection"; for the final test period it was even smaller, about 0.1).

An example from school: the toppers of one test are not always the toppers of the next test. Being at the top once does not guarantee being there again. Late deliveries are the same. A bad week hits lots of sellers, and then it moves on.

**5. Rain looked like a big deal, but it was probably a trick.** When I split the orders into five groups by how much it had rained, the late rate rose from **2.9%** in the driest group to **10.3%** in the wettest. That looks dramatic! But rainy months also happened to be the Black Friday and February-March rush months, and rainy regions are far from the big shipping hubs. So rain might just be standing in for "busy season" and "far away". I decided to test that properly later instead of just believing it.

---

## 6. How the computer learns

A **model** is a rule-maker. I show it thousands of past orders, each with the details known at the time of ordering, plus the answer ("late" or "on time"). It works out which details tend to go with late deliveries, and then it can give a **risk score** to a new order.

The details I give it are called **features**. Examples: how many days were promised, the distance, the price, the seller's recent record, the day of the week, whether a holiday is coming. I ended up using **29** features.

**An important rule: only use what you know at checkout.** I never gave the model anything from after the order was placed, like when the courier collected it, because in real life you would not know that yet. That would be cheating.

### The history trap

Some features describe the past, like "how often was this seller late recently?" I had to be very careful here. For each order, I counted only deliveries that had **finished before that order was placed**. If I accidentally included information from later, the model would be "peeking at the answer key", and its results would look better than they really are.

I did not just trust myself. I wrote a test that picks random orders and recomputes this history **by hand**, the slow way, and checks that it matches. It does.

---

## 7. How I tested it fairly

A fair test is the most important part of any such project.

**A. Test on the future, not on random orders.** If you learn from last year's question papers, the real test is *this year's* paper. So I trained on older orders and tested on **later** ones that the model had never seen.

**B. Wait for slow results.** Here is something I only discovered by studying the data. Late parcels take the longest to arrive. After four weeks, about 5% of all orders were still on the way, but about **60% of the late ones** were still on the way!

Imagine judging a race by only counting runners who have finished. The slow runners are missing, so the results look too good. So I left a **45-day gap** before each test period, and I removed training orders whose outcome was not known yet.

**C. Three groups of orders:**

| Group | Orders placed | Number of orders | Share late | Purpose |
|---|---|---|---|---|
| Training | Jan 2017 – Jan 2018 | 46,461 | 5.4% | the model learns from these |
| Validation | March 2018 | 6,802 | 18.6% | I studied how reliable the scores are |
| Test | mid-May – Aug 2018 | 20,874 | 4.2% | the final exam, used **once** |

Notice that March was a very "late" month and the test months were calm. That makes the exam realistic and a bit hard.

**D. Choosing the model without touching the final exam.** To choose between ideas, I used a practice exam made only from the training period: train on an earlier part, test on a later part, three times. The real test was kept for the end.

I should be honest here. My very first quick experiment did look at the test set once. I did not use it to tune anything. I wrote that down, and then made every real decision using the practice exams only.

---

## 8. The results, in plain words

**The main model reached a score of 0.72** (the technical name is ROC-AUC). Here is what that means: pick one late order and one on-time order at random. About **72 times out of 100**, the model gave the late order the higher risk score. A coin flip would give 50. A perfect model would give 100. So it is clearly better than guessing, but far from perfect.

Another way to see it: if I sort all test orders by risk and look at the riskiest **10%**, that group contains **24% of all the late orders**. A random 10% would contain only 10%.

I also compared it with other ideas:

| Idea | Score (0.5 = guessing) |
|---|---|
| Just look at the seller's past late rate | 0.51 (no better than guessing!) |
| A model using only 5 simple facts: how the promise was set and the calendar | 0.70 |
| **My model (29 features, a "logistic regression")** | **0.72** |
| A more complicated, popular model (called LightGBM) | 0.65 |

Three things surprised me.

**Surprise 1: The simpler model beat the fancier one.** The complicated model scored worse. My guess is that it "memorised" the special weeks (like Black Friday) instead of learning general patterns, like a student who memorises last year's answers and then fails when the questions change. I tried several versions of the complicated model, and they all lost, but I did not try every possible setting, so I cannot say it always loses.

**Surprise 2: Five facts did almost as well as twenty-nine.** The difference between the 5-fact model and my full model was too small to be sure it was real. The lesson: most of the useful information is simply **how the promise was set**: how many days, which route, which time of year.

**Surprise 3: Rain barely mattered.** Remember the dramatic 3% to 10% pattern? Once the model already knew the distance, the promise, and how late things had been recently, adding rainfall improved the score by only about **0.002**, which is tiny. The rain was mostly a decoy for "busy season" and "far away". I collected that data and it gave me a clear answer: *no big effect*. A "no" is still a result, and I left rain out of the final model.

---

## 9. Where the model is weak

**Weakness 1: The scores are only trustworthy when conditions are normal.** For calm months, the model's percentages were close to the truth (it predicted about 4.7% late, and 4.2% really were late). But in the March rush, it predicted about **7%** while the truth was **19%**. It can still tell which orders are riskier than others (the *ordering* works), but the overall *level* is wrong when the world suddenly changes.

I tried to fix this by "adjusting" the scores using March data. It made things **worse** in later months (it predicted 14% when only 4% were late). The reason: March was an unusual month, so what I learned from it did not apply later. So for decisions, I use the **ranking** ("this order is in the riskiest 10%") rather than the exact percentage.

**Weakness 2: It gets worse over time.** Right after training, the score was about 0.75 (May and June). It dropped to 0.67 in July and 0.61 in August. This is like a weather forecast model that was trained last year: the world drifts away from it. In real life, such a model has to be retrained regularly.

A natural question: "Maybe August looks bad because the data got messy at the end?" I checked. The last days of the dataset do look odd (orders suddenly drop from about 250 a day to almost none, which is an artifact of how the data was collected), so I removed those days from the test. But the decline is real: it was not caused by that.

---

## 10. Is it actually worth using? (my favourite part)

A good score does not automatically mean a good business decision. Let me use an example with made-up money.

Suppose that when I flag a risky order, the shop spends **1 unit** on an action (say, sending it by a faster courier), and this action works **half the time**. If a late order costs the shop **10 units** (a bad review, an angry call, a refund), is it worth acting?

The action is worth it only when the chance that an order is late is **at least 1 ÷ (0.5 × 10) = 20%**. My model's top 1% of risky orders are late about 20% of the time, and its top 10% are late only about 10% of the time. So acting on many flagged orders would waste more money than it saves.

What I found:

- If a late order costs **10 times** the action: savings are basically **zero**.
- If a late order costs **20 times** the action: savings are small (about 1 to 2%).
- If a late order costs **40 times** the action (and the action works half the time): flagging about one third of orders **saves about 12%** of the cost.
- If the action is more effective (works 80% of the time), the savings are about 2%, 12% and 36% for 10×, 20× and 40×.

So a 0.72 score is **only useful when the action is cheap compared with the damage of a late order**, for example just showing a more honest delivery date, which costs almost nothing. This is something I would have missed if I had stopped at "the model scores 0.72, great!". These cost numbers are my own assumptions, not measured results. No shop has actually tried this.

---

## 11. From an experiment to something people can use

A model on my laptop helps nobody. So I turned it into a small product with three parts.

**1. An API (a "waiter" for the model).** Another program sends an order's details, and gets back: the chance of being late, how risky it is compared with recent orders, a flag if it is in the riskiest 10%, and **the reasons** (for example, "long distance raises the risk; a long promised time lowers it"). If someone sends nonsense (like a promised date before the order date), it answers politely with an error and does not crash.

**2. Docker (a "lunchbox" with everything packed in).** A computer needs the right software versions to run my model. Docker packs the model and everything it needs into one box that runs the same way on any machine. I built the box, started it, sent it a real order, and it answered correctly.

**3. A web app.** A simple page where you can pick a real order, change details such as the promised date, and see the risk and the reasons. It also shows how good the model is and has a slider for the cost example from Section 10.

**20 automatic tests.** These are small programs that check my work every time. Two of them matter most:

- One checks that the history features are computed using only the past (no peeking).
- One checks that **the model gives the same features when it is running live as when I trained it.** This matters because a classic hidden bug is that "training" and "live" quietly calculate things slightly differently. To make sure this test really works, I deliberately broke one formula by a tiny amount, and the test failed, as it should. Then I fixed it.

---

## 12. Mistakes I made along the way

I think this is the most useful part to share.

- **I looked at the test set once too early.** My very first quick experiment did. I noted it openly and made my later choices in a cleaner way.
- **My first "fancy" model was worse than a simple one.** I did not tune it on the final exam to make it win. I investigated *why* instead (many of my features were growing over time as the business grew, which let a flexible model memorise the calendar), and fixed my features.
- **A chart in my web app was blank.** The tests passed but the chart was empty, because of a small bug in how I fed the numbers in. I only saw it when I opened the page in a real browser. Lesson: tests do not catch everything, so look at the actual thing.
- **I changed the test period and forgot some numbers in the text.** After shortening the test window (to remove those odd last days), a few numbers I had written in the explanations were now slightly out of date. I rechecked every number against the saved results, and caught that one of my own text-edits had even accidentally changed a number. Lesson: write the text from the results file, not from memory.

---

## 13. What this project does *not* show

- It does **not** show this would reduce late deliveries in real life. Nobody tried the actions.
- It is **not** about India, Flipkart, or Meesho. It is 2016–2018 data from Brazil.
- Orders that were **never** delivered are missing from the data, so the true late rate may be higher.
- The model uses the **state** where the customer lives. If a shop used it to give some regions worse service, that would be unfair. It should only be used to *help* risky orders, not to punish any region.
- The signal is **modest**. I did not try every model or every setting, so "the simple model wins" means "here, in my tests", not "always".

---

## 14. What I learned

1. **A fair test matters more than a fancy model.** Most of my effort went into making the test honest.
2. **Simple can beat complicated**, especially when there is not much data and the world keeps changing.
3. **Data that looks exciting can be a decoy.** The rain pattern was dramatic and mostly meaningless.
4. **"A few bad sellers" and "the same bad sellers every time" are different things.**
5. **A model score is not a business decision.** The cost calculation showed that a decent score can still be worth almost nothing.
6. **Writing down failures and limits makes a project more trustworthy**, not less.
7. **Check the real thing, not only the tests.**

---

## Mini dictionary

- **Model:** a program that learns patterns from past examples and gives a score for a new one.
- **Feature:** one fact about an order that the model is allowed to use (distance, days promised, and so on).
- **Label:** the correct answer for a past example ("late" or "on time").
- **Training / test:** training is learning from examples; the test is a fair exam on examples the model has never seen.
- **Data leakage:** accidentally letting the model see information it would not really have, like peeking at the answer key. It makes results look falsely good.
- **ROC-AUC (the 0.72 score):** how often the model gives a higher risk to a late order than to an on-time order. 0.5 is a coin flip; 1.0 is perfect.
- **Calibration:** whether a "20% risk" really means about 20 in 100 such orders are late.
- **Drift:** the world changes, so a model that was good yesterday gets worse.
- **API:** a way for one program to ask another program a question and get an answer.
- **Docker:** a "box" that packs a program and everything it needs so it runs the same everywhere.

*The code, the notebooks, the full numbers and all the limitations are in the project repository: github.com/rajesh22764/delivery-risk-olist.*
