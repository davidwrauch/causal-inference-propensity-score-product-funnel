Causal inference: does clicking a funnel link actually increase conversion?

TLDR;
Users who clicked the link appeared to convert better, but after matching them to similar users who did not click the link, the difference disappeared. The higher conversion rate was driven by who clicked the link, not the link itself.

Many product analytics questions are really causal questions. For example, if users who click a help link in a funnel convert at higher rates, does that mean the link improves conversion, or does it simply attract users who were already more motivated?

This project applies propensity score matching to separate correlation from causation in a product funnel. The analysis evaluates whether clicking a link in a tech company application funnel actually improves downstream revenue per applicant (RPA), or whether the apparent effect is driven by selection bias.

Approach

Users who clicked the link are matched with similar users who did not click the link using k-nearest neighbor matching based on observable characteristics such as time period and user attributes. This creates a control group of users who are as similar as possible to the treated users except for the link click itself.

Once matched pairs are created, the outcome metric (revenue per applicant) is compared between the two groups to estimate the causal impact of the link.

This approach helps avoid a common mistake in funnel analysis: comparing treated users to the entire population, which violates the assumption that the groups are comparable.

Results

A naive comparison suggests that users who click the link have higher revenue per applicant. However, after matching similar users with and without the link click, the difference disappears and is no longer statistically significant.

In other words, the observed relationship appears to be selection bias, not a causal effect.

Product implication

The link introduces substantial drop-off in the funnel but does not appear to improve downstream outcomes once user characteristics are controlled for. Based on this analysis, the link likely adds friction without providing measurable benefit and could reasonably be removed or redesigned.

Why this project matters

This example illustrates a common analytics problem: behavioral signals in product funnels are often confounded by differences in user motivation. Matching methods such as propensity score analysis allow product teams to estimate causal effects even when randomized experiments are not available.

The same approach can be applied to questions such as:

whether specific onboarding steps improve retention

whether certain marketing channels truly drive higher-value users

whether feature usage predicts better outcomes or simply reflects user intent
