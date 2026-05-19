# Churn Prediction System - Visual Diagrams
# Copy these to Mermaid Live Editor: https://mermaid.live/

## Diagram 1: High-Level System Flow

```mermaid
flowchart LR
    subgraph INPUT["📊 DATA SOURCES"]
        A[Historical\nTerminations]
        B[Client\nCommunications]
        C[Health\nReports]
    end

    subgraph PROCESS["🤖 AI ENGINE"]
        D[Feature\nExtraction]
        E[XGBoost\nModel]
        F[Risk\nScoring]
    end

    subgraph OUTPUT["🎯 RESULTS"]
        G[🔴 Critical]
        H[🟠 High]
        I[🟡 Medium]
        J[🟢 Low]
    end

    A --> D
    B --> D
    C --> D
    D --> E
    E --> F
    F --> G
    F --> H
    F --> I
    F --> J

    style G fill:#ff6b6b
    style H fill:#ffa94d
    style I fill:#ffd43b
    style J fill:#69db7c
```

---

## Diagram 2: Decision Flow

```mermaid
flowchart TD
    A[Client in\nSystem] --> B{Run\nPrediction}
    B --> C{Risk\nScore?}
    
    C -->|75-100%| D[🔴 CRITICAL]
    C -->|50-74%| E[🟠 HIGH]
    C -->|25-49%| F[🟡 MEDIUM]
    C -->|0-24%| G[🟢 LOW]
    
    D --> H[Executive Call\nWithin 48hrs]
    E --> I[CSP Outreach\nWithin 1 Week]
    F --> J[Increase\nCheck-ins]
    G --> K[Continue\nStandard Care]
    
    H --> L{Client\nRetained?}
    I --> L
    J --> L
    
    L -->|Yes| M[✅ Success:\nLog & Monitor]
    L -->|No| N[📝 Learn:\nUpdate Model]

    style D fill:#ff6b6b
    style E fill:#ffa94d
    style F fill:#ffd43b
    style G fill:#69db7c
    style M fill:#69db7c
```

---

## Diagram 3: Feature Importance

```mermaid
pie title Top Churn Predictors
    "Seat Count Reduction" : 39
    "No Replacement" : 26
    "Replacement Urgency" : 7
    "Client Tenure" : 7
    "Termination Type" : 5
    "Check-in Frequency" : 5
    "Days Since Check-in" : 4
    "Days Since Communication" : 4
    "Other Factors" : 3
```

---

## Diagram 4: Data Pipeline

```mermaid
flowchart TB
    subgraph SOURCES["Data Sources"]
        A["📋 Google Form\n(Terminations)"]
        B["💾 Supabase\n(Live Data)"]
    end

    subgraph LOCAL["Local Training (Isolated)"]
        C["📥 CSV Import"]
        D["🗄️ SQLite DB\n(1,355 records)"]
        E["⚙️ Feature\nEngineering"]
        F["🎓 Model\nTraining"]
        G["📊 Validation\n(20% holdout)"]
    end

    subgraph PRODUCTION["Production"]
        H["📈 Predictions"]
        I["📱 Dashboard"]
        J["👥 CSP Team"]
    end

    A --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G -->|Approved| H
    B --> H
    H --> I
    I --> J

    style LOCAL fill:#e7f5ff
    style PRODUCTION fill:#d3f9d8
```

---

## Diagram 5: Model Performance Overview

```mermaid
quadrantChart
    title Model Performance Analysis
    x-axis Low Recall --> High Recall
    y-axis Low Precision --> High Precision
    quadrant-1 Ideal Zone
    quadrant-2 Conservative
    quadrant-3 Poor Model
    quadrant-4 Aggressive

    Our Model: [0.786, 0.55]
```

---

## Diagram 6: Confusion Matrix Visual

```mermaid
flowchart TB
    subgraph Reality["ACTUAL OUTCOME"]
        subgraph Stayed["Clients Who Stayed (257)"]
            TN["✅ True Negative\n248"]
            FP["⚠️ False Positive\n9"]
        end
        subgraph Left["Clients Who Left (14)"]
            FN["❌ False Negative\n3"]
            TP["✅ True Positive\n11"]
        end
    end

    subgraph Prediction["MODEL PREDICTION"]
        PredStay["Predicted: Stay"]
        PredLeave["Predicted: Leave"]
    end

    PredStay -.-> TN
    PredStay -.-> FN
    PredLeave -.-> FP
    PredLeave -.-> TP

    style TN fill:#69db7c
    style TP fill:#69db7c
    style FP fill:#ffd43b
    style FN fill:#ff6b6b
```

---

## Diagram 7: Implementation Timeline

```mermaid
gantt
    title Churn Prediction Rollout
    dateFormat  YYYY-MM-DD
    section Phase 1
    Model Training        :done, p1, 2026-04-01, 2026-04-23
    Internal Validation   :done, p2, 2026-04-20, 2026-04-28
    Executive Review      :active, p3, 2026-04-28, 2026-05-05
    section Phase 2
    Pilot with 2 CSPs     :p4, 2026-05-05, 2026-05-19
    Feedback & Adjust     :p5, 2026-05-19, 2026-05-26
    section Phase 3
    Full Team Rollout     :p6, 2026-05-26, 2026-06-09
    Dashboard Training    :p7, 2026-06-02, 2026-06-09
    section Ongoing
    Quarterly Retraining  :p8, 2026-07-01, 2026-07-07
```

---

## How to Use These Diagrams

### Option 1: Mermaid Live Editor
1. Go to https://mermaid.live/
2. Copy any diagram code above
3. Paste into editor
4. Download as PNG/SVG

### Option 2: VS Code Extension
1. Install "Markdown Preview Mermaid Support"
2. Open this file in VS Code
3. Preview to see rendered diagrams

### Option 3: Google Slides
1. Export diagrams as PNG from Mermaid Live
2. Insert images into Google Slides
3. Add text and animations

### Option 4: Notion/Confluence
1. Both support Mermaid natively
2. Paste code blocks directly
3. They render automatically
