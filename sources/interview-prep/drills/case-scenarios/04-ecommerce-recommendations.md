# Case Scenario 4: MercadoPlus — E-Commerce Personalization

## Setup (Read Aloud as Interviewer)

Our client is **MercadoPlus**, a fast-growing e-commerce platform in Latin America with 8 million monthly active users and 500K product SKUs across electronics, fashion, and home goods.

Their conversion rate is 2.3% — below the industry benchmark of 3.5%. They believe the problem is that customers can't find relevant products. Currently, the homepage shows the same trending products to everyone, and search results are sorted by popularity only.

MercadoPlus wants a consulting firm to design a personalization strategy that increases conversion and average order value. They have a 6-month timeline and a $1.5M budget.

---

## Information to Reveal When Asked

**If asked about data:**
- User behavior: browsing history, search queries, cart additions, purchases, returns, time on page
- Product data: category, price, brand, images, descriptions, ratings, reviews
- User profiles: location, device, registration date, purchase frequency
- Contextual: day of week, time, seasonal trends, promotions active

**If asked about current tech stack:**
"Basic MySQL database, Python backend, no ML in production yet. They have a small data team (3 engineers, 1 analyst) but no ML experience."

**If asked about cold start:**
"About 40% of monthly active users are new or have fewer than 3 sessions. Product catalog changes weekly — ~5K new SKUs per month."

**If asked about revenue:**
"Average order value is $45. Monthly revenue is about $8.3M. Top 10% of customers generate 50% of revenue."

---

## The Twist

"The VP of Marketing is pushing hard for a deep learning recommendation engine because 'that's what Amazon uses.' The CTO is concerned about the team's ability to build and maintain it. How would you advise?"

*Expected insight: 80/20 principle. Amazon has thousands of ML engineers — MercadoPlus has 3 data engineers. Start with collaborative filtering or simple content-based filtering. Get to production fast, measure impact, iterate. Deep learning is a year-2 goal if the simpler approach proves the concept.*

---

## What a Strong Answer Covers

1. **Reframe:** "How do we increase conversion from 2.3% to 3.5%+ through personalized product discovery, within a $1.5M budget and 6-month timeline, given the team's current capabilities?"
2. **80/20 approach:** Don't build Netflix's recommendation engine. Start simple, prove value, iterate.
3. **Phased roadmap:**
   - Month 1-2: Content-based filtering (use product attributes to recommend similar items) + "customers who bought X also bought Y" (association rules)
   - Month 3-4: Collaborative filtering for users with purchase history
   - Month 5-6: A/B test, optimize, measure impact
4. **Cold start solution:** New users → content-based (by browsing behavior). New products → attribute-based similarity to existing popular products.
5. **Where to personalize:** Not just homepage. Search ranking, category pages, cart cross-sell, email recommendations, re-engagement for dormant users.
6. **Team constraint is real:** Recommend hiring 1-2 ML engineers or using managed services (SageMaker, Vertex). The 3 existing engineers can maintain a simpler system.
7. **Quantitative impact:** Going from 2.3% to 3.5% conversion = 8M × 1.2% × $45 = $4.3M additional monthly revenue. Against $1.5M investment = payback in 11 days.
