# AORA Take-Home Assessment

## Part 1: Price Estimator (Coding)

**Context:** We have acquired a precision machine shop, and their lead estimator has been quoting aerospace parts by hand for 30 years. He doesn't use a formula; he uses "feel." We've extracted a messy CSV of 500 historical quotes.

### The Task:

Using the provided `aora_historical_quotes.csv`, build a prototype backend service (or a well-documented notebook) that achieves the following:

1. **The Engine:** Create a model/logic that can predict the `TotalPrice_USD` for a new quote.
2. **Feature Extraction:** The `PartDescription` contains "tribal" indicators of complexity. How do you programmatically extract these to improve accuracy?
3. **Bias Detection:** There are three different estimators in the data. Analyze and account for their individual biases. Who is the "safe" quoter? Who is the "aggressive" quoter?
4. **The "Human-in-the-Loop" Workflow:** Design a simple API endpoint where a human can "override" an AI quote. How does your system store this correction to "learn" the missing variable (e.g., specific material hardness or wall thickness)?
5. **Missing Variables:** Aside from the provided data, what variables would you want to ingest to make more accurate quotations? i.e. what would you do if provided a 3D model/renders/measurements of the desired parts.

You can use existing paid APIs, AI, or tools in your solution to the problem; we don't require you to design an ML model from scratch. Prototype an overall solution that works best.

**Data:** [resources/aora_historical_quotes.csv](../resources/aora_historical_quotes.csv)

---

## What We Are Looking For:

- **The "Problem" First:** We value thinking around asking *why* the scrap rate is high before suggesting a $50k Computer Vision system.
- **Flexibility:** Can you handle messy legacy systems.
- **Pragmatism:** In a rollup, "boring" automation that saves 40 hours of fax-entry (Co. E) is often more valuable than a complex AI model that fails in a dusty welding shop (Co. D).
