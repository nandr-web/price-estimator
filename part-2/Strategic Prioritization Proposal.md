
## 1. Day 1: Gather information

Before any site visits, I'd inventory what's available remotely across all five companies:

- **Existing cameras** — Do any sites have surveillance or process monitoring cameras? If so, request access to feeds or recordings. Even low-quality footage may be useful to get a rough idea of workflows, rigs and tooling.
- **Performance records** — Production logs, scrap/reject rates, downtime records, shift reports. If possible, also operator/employee records to identify the best performers in preparation for visits, to know who to observe most.
- **Process and product documentation** — Runbooks, training materials, catalog, diagrams.
- **Regulatory landscape** — Identify applicable regulations per company (especially Co. B's medical device requirements). Need to understand what we can and can't change.

**If no cameras exist:** If possible, I'd install inexpensive cameras, even if the rationale is "for security purposes", unless regulations prohibit it or there's potential worker pushback (keeping good relations with them is probably important). Key cost consideration: wireless signal reliability and wiring logistics may matter more than camera cost itself.

---

## 2. Priority Matrix

I'd spend Month 1 on data gathering across all five companies. I wouldn't commit to one company before understanding the proper context for it.

### Assessment Criteria

- **Impact** — Estimated dollar value of the problem (labor cost, scrap cost, lost throughput, strategic value to the portfolio). Where I lack numbers, I flag what I need.
- **Feasibility** — Technical difficulty, cost of intervention, change management risk, regulatory complexity, and time to first measurable result.

### Rankings

| Company                | Impact                | Feasibility                           | Priority             |
| ---------------------- | --------------------- | ------------------------------------- | -------------------- |
| **E** (RFQ Matching)   | Medium-High           | High                                  | First                |
| **D** (Welding/Scrap)  | High                  | Medium / Unclear                      | Second / Gather data |
| **A** (CNC Downtime)   | Unclear               | High to measure, Low to fix           | Gather data          |
| **B** (Med QC)         | Unclear, possibly low | Unclear (regulations?), possibly high | Gather data          |
| **C** (Stator Winding) | Medium-High           | Low                                   | Delay                |

### Justification

**(RFQ Matching)**
**Co. E first** — Lowest risk of wasted work. Solvable with software and process design. Much of the work (digitization tools, database building) can be done on the move while visiting other sites. 40% of a team's time seems significant (unclear their size or salary, however). Strategically, having a centralized software where we can plug in market intelligence, vendor data, and demand signals benefits the entire collection. Building it for Co. E (RFQ Matching) let's us start laying out the framework for more informed / automated decision making across all companies.

**(Welding/Scrap)**
**Co. D second** — 12% scrap rate on heavy steel fabrication is likely substantial in yen/dollar terms. Thick plate and multi-pass welds are expensive, and I'd assume reducing the scrap rate even by half would justify the investment. The first step (analyzing NCR data) is cheap and high-information. It's unclear what the issues are, but I suspect a combination of (1) processes / training, (2) tooling, and/or (3) scaffolding would be worth investment. (Tooling, for example, in case issues stem from difficult angles; for which we could find cost-effective setups.)

**(CNC Downtime)**
**Co. A — gather data** — Uptime data collection is low-cost and non-invasive (~$500 total, see details below). I'd start collecting immediately but the actual fixes (reducing setup time) require understanding what's driving the downtime first, and the 30-year-old controllers limit what's possible without capital investment.

**(Med QC)**
**Co. B — gather data** — I can't prioritize this until I know whether the paper trail and manual measurement requirements are regulatory mandates or legacy habit. If it's habit, this may be a quick win (CV may not even be necessary, just automated tooling). If it's regulation, I'd delay while we investigate compliant avenues. Importantly, though: this is Quality Control on measurements - the gain of automating this may be low, I need the numbers. On the other hand, if there's risks of fines or litigation for incorrect measurements, I'd upgrade the priority on this one.

**(Stator Winding)**
**Co. C — delay** — High potential long-term value, but the path is long: install sensors → collect months of data → understand patterns → automate. Each step has risk, and workforce cost reduction may face regulatory or cultural barriers. I'd install sensors first if cheap enough and let data accumulate passively while we focus elsewhere. We'd want to analyze the data offline and/or come back when possible to identify the patterns / learn the tribal knowledge.

---

## 3. Month 1 Plan

| Week       | Activity                                                                                                                | Who    |
| ---------- | ----------------------------------------------------------------------------------------------------------------------- | ------ |
| 1          | Remote data gathering across all five companies (records, documentation, regulations, and security footage if possible) | Both   |
| 2          | Co. D site visit (inspect welding process, pull NCR/reject data, identify scrap root causes)                            | Senior |
| 2          | Co. E site visit (shadow sales team, collect sample RFQs, understand workflow)                                          | Junior |
| 3          | Co. C site visit (observe winding, identify tensioner type, assess sensor placement)                                    | Senior |
| 3          | Co. A site visit (assess controllers, identify stack light / power monitoring options)                                  | Junior |
| 4          | Co. B site visit (verify regulatory requirements, assess QC workflow)                                                   | Junior |
| 4          | Planning and presenting proposal                                                                                        | Senior |
| Background | Build initial RFQ digitization prototype for Co. E                                                                      | Both   |

I'd spend 2-5 days per site — long enough to get past the "quick tour" and capture real workflow after people are used to us being there. The visit order is mostly arbitrary (I'd group by proximity in practice), but reflects my initial assessment of risk vs. reward.

After Month 1, we have the data to commit confidently.

---

## 4. Company Details

### Co. E — RFQ Matching

**Risks:** OCR on Japanese manufacturing documents (Kanji + technical drawings) is non-trivial — need to understand the actual RFQ formats first. We also need to consider drawings / schematics if applicable. Bigger risk is change management: salespeople's value is "I know the network." If we alienate them, they take their knowledge and connections with them.

**Approach:** Build a central system that serves the salespeople first and captures intelligence as a byproduct:
1. **RFQ digitization** — Do a quick search for off-the-shelf solutions, but more likely develop our own solution (cheaper and fast nowadays), iterating with the team, measure speed gains per step (intake vs. matching vs. quoting).
2. **Shop outreach tracker** — Salespeople log contacts and responses per RFQ. Framed as their state-tracking tool; simultaneously generates capacity signals and pricing intelligence.
3. **Shop capability database** — Pre-populate from public info. Salespeople refine it as the experts, not the subjects. Protect the relationship layer as their competitive edge.
4. **Quote history + suggested matches** — Searchable archive of past quotes; eventually, suggest top shops with reasoning. Salesperson always decides.

The key design principle: every data capture point should help the salesperson do their job.

---

### Co. D — Welding / Scrap

Note: Full automation is likely not cost-effective / feasible. As the instructions mention, the environment here makes CV challenging. There may be solutions for it, but they are probably expensive.

**Risks:** The 12% likely has multiple root causes: materials, process parameters, operator skill, fit-up, and environment.

**Approach:** Pull NCR/reject data first and categorize by defect type — that tells us where to focus. Porosity → gas/contamination; cracking → preheat control; incomplete fusion → technique/parameters; distortion → fixturing/sequencing. The fix may be procedural (checklists, parameter standardization, welder feedback loops) before it's technological.

---

### Co. A — CNC Downtime (Hardware/Software Bridge)

**Risks:** We don't know actual downtime figures. The 30-year-old controllers limit what's fixable without capital investment — data collection is cheap, but reducing setup time may not be.

**Approach:** Non-invasive, external monitoring — we don't touch the controllers.

| Method | How it works | Cost/machine | Data quality |
|--------|-------------|--------------|--------------|
| CT clamp on spindle motor | Clip-on current transformer on power cable. Power draw = cutting. Logs to Raspberry Pi/ESP32. | ~$20-50 + MCU | High — clean binary signal with timestamps |
| Stack light tap | Wire into existing status light circuit. | ~$30-50 + MCU | High — multi-state (running, idle, alarm) |
| Vibration sensor | Accelerometer on machine frame. | ~$20-40 + MCU | Medium — needs calibration |

Start with CT clamps, add a camera per area for context on *why* machines are idle. Total cost: under $500. The data tells us whether the problem is setup, scheduling, or maintenance — before we invest in solutions.

---

### Co. B — Med QC

**Risks:** Regulatory gate — I need to verify whether manual measurement and handwritten paper trails are mandated or legacy habit. This changes feasibility entirely. Also, 20% of QC time sounds meaningful, but without team size/cost figures, the dollar value could be low. Worth noting: if there's risk of fines or litigation for incorrect measurements, that escalates the priority.

**Approach:** If paper trail is habit → digital logging at QC stations (simple, fast payback). If regulatory → compliant electronic QMS (heavier lift, may not fit our 6-month window). Either way, the bottleneck is likely documentation overhead, not the measurement itself — digitizing logging may capture most of the value even if measurement stays manual.

---

### Co. C — Stator Winding

**Risks:** Long path to automation with sunk cost risk at each step. Tribal knowledge runs deeper than just tension — operators rely on touch (vibration, drag), sound (wire "singing," motor strain), vision (coil neatness, slot fill), and timing (winding rhythm). Tension sensors alone won't capture the full picture. Tensioner type (mechanical vs. magnetic) determines retrofit approach — must verify on-site.

**Approach:** Install sensors during a follow up site visit if cheap (tension, vibration/mic, camera, RPM). Let data accumulate - we can do offline learning. Near-term value: sensor data as a training aid for new operators — making the "feel" visible could relieve the bottleneck before full automation.