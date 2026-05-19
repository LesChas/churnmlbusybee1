# Churn Prediction Model
## How It Works (Non-Technical Overview)

---

## The Big Picture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                        🎯 CHURN PREDICTION SYSTEM                           │
│                                                                             │
│    "Learning from the past to protect our future"                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

     STEP 1                    STEP 2                    STEP 3
    LEARNING                 ANALYZING                 ALERTING
        │                        │                        │
        ▼                        ▼                        ▼
┌───────────────┐        ┌───────────────┐        ┌───────────────┐
│               │        │               │        │               │
│  📚 HISTORY   │   →    │  🧠 AI MODEL  │   →    │  🚨 DASHBOARD │
│               │        │               │        │               │
│  "What did    │        │  "Find the    │        │  "Show me     │
│   clients do  │        │   patterns"   │        │   who's at    │
│   before they │        │               │        │   risk NOW"   │
│   left?"      │        │               │        │               │
│               │        │               │        │               │
└───────────────┘        └───────────────┘        └───────────────┘
```

---

## Step 1: Learning from History

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   📊 WHAT WE COLLECTED                                                      │
│   ────────────────────                                                      │
│                                                                             │
│   Over the years, we've recorded 1,355 team member terminations.           │
│   Each record tells us:                                                     │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   👤 Which CLIENT was involved                                      │  │
│   │   📉 Did they REDUCE their team size after?                         │  │
│   │   🔄 Did they want a REPLACEMENT?                                   │  │
│   │   📅 How LONG had they been a client?                               │  │
│   │   📞 When was the last CHECK-IN?                                    │  │
│   │   💬 Were there COMMUNICATION gaps?                                 │  │
│   │   ❌ Did the client eventually LEAVE entirely?                      │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   Of these 1,355 records, 69 clients (5.1%) ended up terminating           │
│   their entire relationship with us.                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Step 2: The AI Finds Patterns

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   🧠 WHAT THE AI DISCOVERED                                                 │
│   ─────────────────────────                                                 │
│                                                                             │
│   Think of it like a detective reviewing 1,355 case files and asking:      │
│   "What did clients who LEFT have in common?"                              │
│                                                                             │
│   The AI found these WARNING SIGNS:                                        │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   #1  📉 SHRINKING TEAMS (39% of the signal)                        │  │
│   │       When clients reduce seat count, they're 4x more likely        │  │
│   │       to leave entirely. They're "pulling back."                    │  │
│   │                                                                     │  │
│   │   #2  ❌ NO REPLACEMENT NEEDED (26% of the signal)                  │  │
│   │       When a TM leaves and the client says "don't replace them,"   │  │
│   │       it's a major red flag.                                        │  │
│   │                                                                     │  │
│   │   #3  🆕 NEW CLIENTS (7% of the signal)                             │  │
│   │       Clients in their first 1-2 years are more vulnerable.        │  │
│   │       They haven't built deep loyalty yet.                          │  │
│   │                                                                     │  │
│   │   #4  📞 GOING QUIET (5% of the signal)                             │  │
│   │       Less communication = trouble brewing.                         │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   The AI doesn't just look at one factor—it weighs ALL of them together.  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Step 3: Scoring Current Clients

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   🎯 HOW THE SCORE WORKS                                                    │
│   ──────────────────────                                                    │
│                                                                             │
│   Now the AI looks at CURRENT clients and asks:                            │
│   "How similar are they to clients who LEFT in the past?"                  │
│                                                                             │
│                                                                             │
│        CLIENT A                              CLIENT B                       │
│   ┌─────────────────┐                   ┌─────────────────┐                │
│   │ • Reduced seats │                   │ • Stable seats  │                │
│   │ • No replacement│                   │ • Always wants  │                │
│   │ • 8 months old  │                   │   replacements  │                │
│   │ • Last call: 6  │                   │ • 3 years old   │                │
│   │   weeks ago     │                   │ • Monthly calls │                │
│   └────────┬────────┘                   └────────┬────────┘                │
│            │                                     │                          │
│            ▼                                     ▼                          │
│   ┌─────────────────┐                   ┌─────────────────┐                │
│   │                 │                   │                 │                │
│   │  RISK: 78% 🔴   │                   │  RISK: 12% 🟢   │                │
│   │                 │                   │                 │                │
│   │  "Looks like    │                   │  "Looks like    │                │
│   │   clients who   │                   │   clients who   │                │
│   │   LEFT"         │                   │   STAYED"       │                │
│   │                 │                   │                 │                │
│   └─────────────────┘                   └─────────────────┘                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## The Complete Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                         COMPLETE SYSTEM FLOW                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║   📋 GOOGLE FORM                                                  ║
    ║   (Team Member Termination Form)                                  ║
    ║                                                                   ║
    ║   Filled out every time a TM leaves a client                     ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
                                    │
                                    │ Data flows automatically
                                    ▼
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║   📊 TRAINING DATABASE                                            ║
    ║   (Secure, Isolated Storage)                                      ║
    ║                                                                   ║
    ║   • 1,355 historical records                                     ║
    ║   • No personal information stored                               ║
    ║   • Only patterns and numbers                                    ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
                                    │
                                    │ AI learns patterns
                                    ▼
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║   🧠 AI MODEL (XGBoost)                                           ║
    ║   "The Pattern Detector"                                          ║
    ║                                                                   ║
    ║   • Trained on historical data                                   ║
    ║   • 96.7% accuracy in ranking risk                               ║
    ║   • Updates quarterly with new data                              ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
                                    │
                                    │ Scores each client
                                    ▼
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║   📱 BUSYBEE DASHBOARD                                            ║
    ║   (What Your Team Sees)                                           ║
    ║                                                                   ║
    ║   ┌───────────────────────────────────────────────────────────┐  ║
    ║   │  CLIENT          │  RISK   │  REASON           │  ACTION  │  ║
    ║   │─────────────────────────────────────────────────────────────│  ║
    ║   │  Acme Corp       │  🔴 78% │  Reduced seats    │  CALL    │  ║
    ║   │  Widget Inc      │  🟠 52% │  No replacement   │  REVIEW  │  ║
    ║   │  TechStart LLC   │  🟢 15% │  Healthy          │  MONITOR │  ║
    ║   └───────────────────────────────────────────────────────────┘  ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
                                    │
                                    │ Team takes action
                                    ▼
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║   👥 CSP TEAM ACTION                                              ║
    ║                                                                   ║
    ║   🔴 Critical (75%+)  →  Executive reaches out within 48 hours   ║
    ║   🟠 High (50-74%)    →  CSP schedules check-in this week        ║
    ║   🟡 Medium (25-49%)  →  Increase touchpoint frequency           ║
    ║   🟢 Low (0-24%)      →  Continue normal engagement              ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
                                    │
                                    │ Outcome tracked
                                    ▼
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║   🔄 FEEDBACK LOOP                                                ║
    ║                                                                   ║
    ║   Did the intervention work?                                      ║
    ║   → Yes: Model learns what "save" looks like                     ║
    ║   → No:  Model learns from the miss                              ║
    ║                                                                   ║
    ║   Every quarter, the model retrains with new data                ║
    ║   to get smarter over time.                                      ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
```

---

## What Makes This Different

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   BEFORE (Gut Feeling)              AFTER (Data-Driven)                    │
│   ────────────────────              ────────────────────                   │
│                                                                             │
│   ❌ "I think they might           ✅ "The data shows 78% risk             │
│      be unhappy"                       because of X, Y, Z"                 │
│                                                                             │
│   ❌ React to termination           ✅ Act weeks BEFORE termination        │
│      notice                                                                 │
│                                                                             │
│   ❌ Every client treated           ✅ High-risk clients get               │
│      the same                          priority attention                   │
│                                                                             │
│   ❌ No record of why               ✅ Every prediction logged             │
│      we lost clients                   with reasons                         │
│                                                                             │
│   ❌ Rely on CSP memory             ✅ System never forgets                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Simple Analogy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   🏥 THINK OF IT LIKE A HEALTH CHECK-UP                                    │
│   ─────────────────────────────────────                                     │
│                                                                             │
│                                                                             │
│   DOCTOR'S APPROACH                    OUR APPROACH                        │
│   ─────────────────                    ────────────                        │
│                                                                             │
│   📋 Reviews your medical history      📋 Reviews client history           │
│                                                                             │
│   🔬 Checks vital signs:               🔬 Checks "vital signs":            │
│      • Blood pressure                     • Seat count changes             │
│      • Heart rate                         • Replacement requests           │
│      • Cholesterol                        • Communication frequency        │
│                                                                             │
│   📊 Compares to population data       📊 Compares to clients who left    │
│                                                                             │
│   ⚠️ Flags risk factors:              ⚠️ Flags risk factors:              │
│      "Your cholesterol is high,           "This client reduced seats,      │
│       you're at higher risk               they're at higher risk           │
│       for heart disease"                  of leaving"                      │
│                                                                             │
│   💊 Recommends preventive action      💊 Recommends proactive outreach   │
│                                                                             │
│   The AI is like an automated health screening                             │
│   for every client relationship, running 24/7.                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Numbers

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   📈 MODEL PERFORMANCE                                                      │
│   ────────────────────                                                      │
│                                                                             │
│                                                                             │
│        ┌─────────────────────────────────────────────────────────┐         │
│        │                                                         │         │
│        │                      96.7%                              │         │
│        │                                                         │         │
│        │   How often the model correctly ranks a risky client   │         │
│        │   higher than a healthy client                          │         │
│        │                                                         │         │
│        └─────────────────────────────────────────────────────────┘         │
│                                                                             │
│                                                                             │
│        ┌───────────────┐    ┌───────────────┐    ┌───────────────┐         │
│        │               │    │               │    │               │         │
│        │     95.6%     │    │     78.6%     │    │      79%      │         │
│        │               │    │               │    │               │         │
│        │   Overall     │    │   Detection   │    │   Of at-risk  │         │
│        │   Accuracy    │    │   Rate        │    │   clients we  │         │
│        │               │    │               │    │   catch early │         │
│        └───────────────┘    └───────────────┘    └───────────────┘         │
│                                                                             │
│                                                                             │
│   In plain English:                                                        │
│   • We get it right 96 times out of 100                                    │
│   • We catch 8 out of 10 at-risk clients before they leave               │
│   • The 2 we miss are room for improvement                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Questions Stakeholders Might Ask

| Question | Answer |
|----------|--------|
| **"Can anyone see this data?"** | No. Only authorized team members with dashboard access. |
| **"Does it store personal information?"** | No. The model only uses patterns (numbers), not names or PII. |
| **"What if the model is wrong?"** | Humans always make the final decision. AI recommends, people decide. |
| **"How often is it updated?"** | The model retrains quarterly with new data. |
| **"Can a client game the system?"** | Clients don't see their scores. This is internal only. |
| **"What does it cost?"** | Training runs locally. No expensive cloud APIs required. |
| **"Who built this?"** | Internal development using industry-standard tools (XGBoost). |

---

## One-Page Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                   🎯 CHURN PREDICTION IN ONE PAGE                           │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   WHAT IT IS:                                                              │
│   An AI system that predicts which clients are at risk of leaving,        │
│   BEFORE they give notice.                                                 │
│                                                                             │
│   HOW IT WORKS:                                                            │
│   1. Learns from 1,355 historical termination records                     │
│   2. Identifies patterns (seat reductions, communication gaps)            │
│   3. Scores each current client from 0-100% risk                          │
│   4. Displays on dashboard with recommended actions                        │
│                                                                             │
│   WHY IT MATTERS:                                                          │
│   • Early warning = time to intervene                                      │
│   • Prioritize high-risk clients                                           │
│   • Protect $150,000+ revenue annually                                     │
│                                                                             │
│   WHAT IT'S NOT:                                                           │
│   • Not a replacement for human judgment                                   │
│   • Not a guarantee (79% detection, not 100%)                              │
│   • Not a "black box" (every score is explainable)                        │
│                                                                             │
│   STATUS:                                                                  │
│   ✅ Model trained and validated                                          │
│   ✅ 96.7% discrimination accuracy                                         │
│   ⏳ Dashboard integration pending                                         │
│   ⏳ Pilot rollout planned                                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
