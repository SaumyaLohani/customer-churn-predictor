import { useState, useMemo } from "react";

const MODEL = {
  featureNames: [
    "tenure", "MonthlyCharges", "TotalCharges",
    "Contract_One year", "Contract_Two year",
    "InternetService_Fiber optic", "InternetService_No",
    "PaymentMethod_Credit card (automatic)", "PaymentMethod_Electronic check", "PaymentMethod_Mailed check",
    "PaperlessBilling_Yes", "SeniorCitizen_1",
  ],
  mean: [31.2, 65.42384615384616, 2217.340897435897, 0.20256410256410257, 0.2205128205128205, 0.4564102564102564, 0.20256410256410257, 0.2128205128205128, 0.3282051282051282, 0.23076923076923078, 0.5897435897435898, 0.1641025641025641],
  scale: [24.1169373401074, 29.712842264978807, 2200.9423469919716, 0.40191029710185616, 0.414592470387851, 0.49809631021899814, 0.40191029710185616, 0.40930177393132167, 0.46955992378500916, 0.42132504423474315, 0.491880156237202, 0.3703686171336982],
  coef: [-0.3082232245224064, 0.031069165029010033, -0.4551506015039453, -0.6686260232696316, -0.9802850063750321, 0.5125077266415867, -0.40849953879580236, -0.09445698815746144, 0.19132824185913413, 0.10130544144944882, 0.17220699875771295, 0.03555380659314459],
  intercept: -0.8256833790990151,
  metrics: {
    accuracy: 0.6938775510204082,
    precisionChurn: 0.4444444444444444,
    recallChurn: 0.8,
    f1Churn: 0.5714285714285714,
    testSetSize: 98,
  },
};

const CONTRACT_OPTIONS = ["Month-to-month", "One year", "Two year"];
const INTERNET_OPTIONS = ["DSL", "Fiber optic", "No"];
const PAYMENT_OPTIONS = ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"];

function defaultCustomer() {
  return {
    tenure: 28,
    MonthlyCharges: 72.6,
    TotalCharges: 1374,
    Contract: "Month-to-month",
    InternetService: "Fiber optic",
    PaymentMethod: "Electronic check",
    PaperlessBilling: true,
    SeniorCitizen: false,
  };
}

function toFeatureValues(c) {
  return {
    tenure: c.tenure,
    MonthlyCharges: c.MonthlyCharges,
    TotalCharges: c.TotalCharges,
    "Contract_One year": c.Contract === "One year" ? 1 : 0,
    "Contract_Two year": c.Contract === "Two year" ? 1 : 0,
    "InternetService_Fiber optic": c.InternetService === "Fiber optic" ? 1 : 0,
    "InternetService_No": c.InternetService === "No" ? 1 : 0,
    "PaymentMethod_Credit card (automatic)": c.PaymentMethod === "Credit card (automatic)" ? 1 : 0,
    "PaymentMethod_Electronic check": c.PaymentMethod === "Electronic check" ? 1 : 0,
    "PaymentMethod_Mailed check": c.PaymentMethod === "Mailed check" ? 1 : 0,
    PaperlessBilling_Yes: c.PaperlessBilling ? 1 : 0,
    SeniorCitizen_1: c.SeniorCitizen ? 1 : 0,
  };
}

function sigmoid(z) {
  return 1 / (1 + Math.exp(-z));
}

function predict(customer) {
  const values = toFeatureValues(customer);
  let z = MODEL.intercept;
  MODEL.featureNames.forEach((name, i) => {
    const scaled = (values[name] - MODEL.mean[i]) / MODEL.scale[i];
    z += MODEL.coef[i] * scaled;
  });
  return { pChurn: sigmoid(z) };
}

function riskBand(pChurn) {
  if (pChurn < 0.3) return { label: "Low", color: "#059669" };
  if (pChurn < 0.6) return { label: "Elevated", color: "#D97706" };
  return { label: "High", color: "#DC2626" };
}

const FRIENDLY = {
  tenure: "tenure",
  MonthlyCharges: "monthly charges",
  TotalCharges: "total charges to date",
  "Contract_One year": "being on a one-year contract",
  "Contract_Two year": "being on a two-year contract",
  "InternetService_Fiber optic": "having fiber internet",
  "InternetService_No": "having no internet service",
  "PaymentMethod_Credit card (automatic)": "paying by automatic credit card",
  "PaymentMethod_Electronic check": "paying by electronic check",
  "PaymentMethod_Mailed check": "paying by mailed check",
  PaperlessBilling_Yes: "paperless billing",
  SeniorCitizen_1: "being a senior citizen",
};

function buildExplanation(customer, pChurn, band) {
  const values = toFeatureValues(customer);
  const contributions = MODEL.featureNames.map((key, i) => {
    const scaledDiff = (values[key] - MODEL.mean[i]) / MODEL.scale[i];
    const contribution = MODEL.coef[i] * scaledDiff;
    return { key, contribution };
  });

  const towardChurn = [...contributions].sort((a, b) => b.contribution - a.contribution).slice(0, 3).filter((f) => f.contribution > 0.04);
  const towardStay = [...contributions].sort((a, b) => a.contribution - b.contribution).slice(0, 3).filter((f) => f.contribution < -0.04);

  const churnDrivers = towardChurn.map((f) => FRIENDLY[f.key] || f.key);
  const retentionFactors = towardStay.map((f) => FRIENDLY[f.key] || f.key);

  let body;
  if (band.label === "Low") {
    body = `This ${(pChurn * 100).toFixed(1)}% score is low churn risk. This customer's profile looks stable${retentionFactors.length ? `, largely thanks to ${retentionFactors.join(" and ")}` : ""}.`;
  } else if (band.label === "High") {
    body = `This ${(pChurn * 100).toFixed(1)}% score is high churn risk. The strongest signals pushing it up are ${churnDrivers.join(" and ") || "several factors above typical levels"}. A proactive retention offer could make sense here.`;
  } else {
    body = `This ${(pChurn * 100).toFixed(1)}% score sits in an elevated-but-uncertain range.${churnDrivers.length ? ` ${churnDrivers.join(" and ")} push it toward churn,` : ""}${retentionFactors.length ? ` while ${retentionFactors.join(" and ")} pull it back toward retention.` : ""}`;
  }

  return `${body} Generated directly from the trained model's own coefficients \u2014 one advantage of an interpretable model like logistic regression over a black box. The model favors recall over precision: it would rather flag a customer who ends up staying than miss one who's about to leave, since a missed churner is far costlier than one unnecessary retention offer. Portfolio demo on a public research dataset, not a production system.`;
}

// ---- Dashboard shell palette: dark control rail + light workspace ----
const RAIL = "#111827";
const RAIL_LINE = "#1F2937";
const RAIL_TEXT = "#E5E7EB";
const RAIL_MUTED = "#9CA3AF";
const RAIL_ACCENT = "#38BDF8";

const CANVAS = "#F3F4F6";
const INK = "#111827";
const INK_SOFT = "#6B7280";
const BORDER = "#E5E7EB";
const PANEL = "#FFFFFF";

const FONT = "ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif";
const MONO = "ui-monospace, 'SF Mono', Menlo, monospace";

function RailField({ label, children, help }) {
  return (
    <div className="mb-5">
      <label className="text-[11px] uppercase tracking-wide block mb-2 font-medium" style={{ color: RAIL_MUTED, fontFamily: FONT }}>
        {label}
      </label>
      {children}
      {help && (
        <p style={{ color: RAIL_MUTED, fontFamily: FONT }} className="text-[11px] mt-1.5">
          {help}
        </p>
      )}
    </div>
  );
}

function Gauge({ pChurn, color }) {
  const size = 200;
  const stroke = 14;
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const pct = Math.min(Math.max(pChurn, 0), 1);
  const offset = c * (1 - pct);
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={BORDER} strokeWidth={stroke} />
      <circle
        cx={size / 2} cy={size / 2} r={r} fill="none" stroke={color} strokeWidth={stroke}
        strokeDasharray={c} strokeDashoffset={offset} strokeLinecap="round"
        transform={`rotate(-90 ${size / 2} ${size / 2})`}
        style={{ transition: "stroke-dashoffset 0.3s ease" }}
      />
      <text x="50%" y="46%" textAnchor="middle" style={{ fontFamily: MONO, fontSize: "34px", fontWeight: 700, fill: INK }}>
        {(pChurn * 100).toFixed(0)}%
      </text>
      <text x="50%" y="62%" textAnchor="middle" style={{ fontFamily: FONT, fontSize: "11px", fill: INK_SOFT, letterSpacing: "0.05em" }}>
        CHURN RISK
      </text>
    </svg>
  );
}

function KpiTile({ label, value, sub }) {
  return (
    <div style={{ background: PANEL, border: `1px solid ${BORDER}`, borderRadius: 10 }} className="p-4">
      <p style={{ color: INK_SOFT, fontFamily: FONT }} className="text-xs mb-1">{label}</p>
      <p style={{ color: INK, fontFamily: MONO }} className="text-2xl font-semibold">{value}</p>
      {sub && <p style={{ color: INK_SOFT, fontFamily: FONT }} className="text-[11px] mt-0.5">{sub}</p>}
    </div>
  );
}

export default function ChurnPredictor() {
  const [customer, setCustomer] = useState(defaultCustomer());
  const [explanation, setExplanation] = useState("");

  const { pChurn } = useMemo(() => predict(customer), [customer]);
  const band = riskBand(pChurn);

  function update(key, val) {
    setCustomer((prev) => ({ ...prev, [key]: val }));
    setExplanation("");
  }

  function handleExplain() {
    setExplanation(buildExplanation(customer, pChurn, band));
  }

  const selectStyle = {
    width: "100%",
    background: "#1F2937",
    border: `1px solid ${RAIL_LINE}`,
    borderRadius: 6,
    padding: "8px 10px",
    fontFamily: FONT,
    fontSize: "0.8125rem",
    color: RAIL_TEXT,
  };

  return (
    <div style={{ background: CANVAS, minHeight: "100%", fontFamily: FONT }} className="w-full flex flex-col md:flex-row">
      {/* Control rail */}
      <div style={{ background: RAIL, borderRight: `1px solid ${RAIL_LINE}` }} className="w-full md:w-[300px] md:min-h-screen shrink-0 p-6">
        <div className="flex items-center gap-2 mb-8">
          <div style={{ background: RAIL_ACCENT, borderRadius: 4 }} className="w-6 h-6 flex items-center justify-center">
            <span style={{ fontFamily: MONO, fontSize: 12, fontWeight: 700, color: "#0B1220" }}>C</span>
          </div>
          <div>
            <p style={{ color: RAIL_TEXT, fontFamily: FONT }} className="text-sm font-semibold leading-none">Churn Console</p>
            <p style={{ color: RAIL_MUTED, fontFamily: FONT }} className="text-[10px] uppercase tracking-wide mt-0.5">Portfolio demo</p>
          </div>
        </div>

        <RailField label={`Tenure — ${customer.tenure} mo`} help="Months as a customer">
          <input type="range" min={0} max={72} step={1} value={customer.tenure}
            onChange={(e) => update("tenure", Number(e.target.value))}
            style={{ accentColor: RAIL_ACCENT, width: "100%" }} />
        </RailField>

        <RailField label={`Monthly charges — $${customer.MonthlyCharges.toFixed(2)}`}>
          <input type="range" min={18} max={120} step={0.5} value={customer.MonthlyCharges}
            onChange={(e) => update("MonthlyCharges", Number(e.target.value))}
            style={{ accentColor: RAIL_ACCENT, width: "100%" }} />
        </RailField>

        <RailField label={`Total charges — $${customer.TotalCharges.toFixed(0)}`}>
          <input type="range" min={0} max={8500} step={25} value={customer.TotalCharges}
            onChange={(e) => update("TotalCharges", Number(e.target.value))}
            style={{ accentColor: RAIL_ACCENT, width: "100%" }} />
        </RailField>

        <RailField label="Contract type">
          <select style={selectStyle} value={customer.Contract} onChange={(e) => update("Contract", e.target.value)}>
            {CONTRACT_OPTIONS.map((o) => <option key={o} value={o}>{o}</option>)}
          </select>
        </RailField>

        <RailField label="Internet service">
          <select style={selectStyle} value={customer.InternetService} onChange={(e) => update("InternetService", e.target.value)}>
            {INTERNET_OPTIONS.map((o) => <option key={o} value={o}>{o}</option>)}
          </select>
        </RailField>

        <RailField label="Payment method">
          <select style={selectStyle} value={customer.PaymentMethod} onChange={(e) => update("PaymentMethod", e.target.value)}>
            {PAYMENT_OPTIONS.map((o) => <option key={o} value={o}>{o}</option>)}
          </select>
        </RailField>

        <div className="flex flex-col gap-2 mt-2">
          <label className="text-xs flex items-center gap-2" style={{ color: RAIL_TEXT, fontFamily: FONT }}>
            <input type="checkbox" checked={customer.PaperlessBilling} onChange={(e) => update("PaperlessBilling", e.target.checked)} />
            Paperless billing
          </label>
          <label className="text-xs flex items-center gap-2" style={{ color: RAIL_TEXT, fontFamily: FONT }}>
            <input type="checkbox" checked={customer.SeniorCitizen} onChange={(e) => update("SeniorCitizen", e.target.checked)} />
            Senior citizen
          </label>
        </div>
      </div>

      {/* Workspace */}
      <div className="flex-1 p-6 md:p-10">
        <div className="max-w-3xl mx-auto">
          <header className="mb-6">
            <h1 style={{ color: INK, fontFamily: FONT }} className="text-2xl font-bold tracking-tight mb-1.5">
              Customer risk overview
            </h1>
            <p style={{ color: INK_SOFT, fontFamily: FONT }} className="text-sm max-w-lg leading-relaxed">
              Logistic regression trained on the IBM/Kaggle Telco Customer Churn dataset. Adjust
              the console on the left; the readout below updates instantly.
            </p>
          </header>

          <div className="grid md:grid-cols-5 gap-5">
            {/* Gauge card */}
            <div style={{ background: PANEL, border: `1px solid ${BORDER}`, borderRadius: 12 }} className="md:col-span-2 p-6 flex flex-col items-center justify-center text-center">
              <Gauge pChurn={pChurn} color={band.color} />
              <span
                style={{ color: band.color, background: `${band.color}18`, fontFamily: FONT }}
                className="text-xs font-semibold px-3 py-1 mt-3 rounded-full"
              >
                {band.label} risk
              </span>
            </div>

            {/* KPI grid + explanation */}
            <div className="md:col-span-3 flex flex-col gap-4">
              <div className="grid grid-cols-2 gap-3">
                <KpiTile label="Model accuracy" value={`${(MODEL.metrics.accuracy * 100).toFixed(1)}%`} />
                <KpiTile label="Recall on churn" value={`${(MODEL.metrics.recallChurn * 100).toFixed(1)}%`} sub="priority metric" />
                <KpiTile label="Precision on churn" value={`${(MODEL.metrics.precisionChurn * 100).toFixed(1)}%`} />
                <KpiTile label="Test set size" value={MODEL.metrics.testSetSize} sub="customers" />
              </div>

              <div style={{ background: PANEL, border: `1px solid ${BORDER}`, borderRadius: 12 }} className="p-5 flex-1">
                <div className="flex items-center justify-between mb-3">
                  <p style={{ color: INK, fontFamily: FONT }} className="text-sm font-semibold">Why this score?</p>
                  <button
                    onClick={handleExplain}
                    style={{ background: INK, color: "#fff", fontFamily: FONT, borderRadius: 6 }}
                    className="text-xs font-medium px-3 py-1.5 hover:opacity-85 transition-opacity"
                  >
                    Explain
                  </button>
                </div>
                {explanation ? (
                  <p style={{ fontFamily: FONT, color: INK, lineHeight: 1.6 }} className="text-sm whitespace-pre-wrap">
                    {explanation}
                  </p>
                ) : (
                  <p style={{ fontFamily: FONT, color: INK_SOFT }} className="text-sm">
                    Click "Explain" for a plain-language breakdown generated straight from the
                    model's own coefficients — no external API call.
                  </p>
                )}
              </div>
            </div>
          </div>

          <div style={{ background: "#EFF6FF", border: `1px solid ${BORDER}`, borderRadius: 10 }} className="text-xs px-4 py-3 mt-5" >
            <span style={{ fontFamily: FONT, color: INK_SOFT }}>
              Portfolio demo on a public research dataset — the same interpretable-model pattern
              applies to any subscription business: SaaS, telecom, fintech, streaming.
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
