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

// ---- Turn a friendly customer object into the one-hot vector the model
// was trained on.
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
  const pChurn = sigmoid(z);
  return { pChurn, values };
}

function riskBand(pChurn) {
  if (pChurn < 0.3) return { label: "Low", color: "#3F6357" };
  if (pChurn < 0.6) return { label: "Elevated", color: "#B07A2E" };
  return { label: "High", color: "#A8412F" };
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

  return `${body} This explanation is generated directly from the trained model's own coefficients \u2014 one advantage of an interpretable model like logistic regression over a black box. The model is tuned to favor recall over precision: it would rather flag a customer who ends up staying than miss one who's about to leave, since a missed churner is far costlier than one unnecessary retention offer. Reminder: this is a portfolio demo on a public research dataset, not a production system.`;
}

const INK = "#16232B";
const INK_SOFT = "#4B5A61";
const BG = "#F2F1EC";
const PANEL = "#FFFFFF";
const LINE = "#DDDAD1";
const ACCENT = "#3F6357";

function Field({ label, children, help }) {
  return (
    <div>
      <label className="text-sm block mb-1" style={{ fontFamily: "ui-sans-serif, system-ui", color: INK }}>
        {label}
      </label>
      {children}
      {help && (
        <p style={{ color: INK_SOFT, fontFamily: "ui-sans-serif, system-ui" }} className="text-xs mt-1">
          {help}
        </p>
      )}
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
    border: `1px solid ${LINE}`,
    padding: "6px 8px",
    fontFamily: "ui-sans-serif, system-ui",
    fontSize: "0.875rem",
    background: "#fff",
    color: INK,
  };

  return (
    <div style={{ background: BG, minHeight: "100%", color: INK, fontFamily: "'Iowan Old Style', 'Palatino Linotype', Georgia, serif" }} className="w-full p-4 md:p-8">
      <div className="max-w-4xl mx-auto">
        <header className="mb-6">
          <p style={{ color: INK_SOFT, fontFamily: "ui-monospace, monospace", letterSpacing: "0.02em" }} className="text-xs mb-2">
            AI/ML PORTFOLIO PROJECT — CUSTOMER RETENTION
          </p>
          <h1 className="text-3xl md:text-4xl mb-2" style={{ fontWeight: 600 }}>
            Customer churn predictor
          </h1>
          <p style={{ color: INK_SOFT, fontFamily: "ui-sans-serif, system-ui" }} className="text-sm max-w-xl leading-relaxed">
            A logistic regression model trained on the IBM/Kaggle Telco Customer Churn dataset.
            Adjust a customer's profile to see the model respond in real time, then ask it to
            explain its own reasoning.
          </p>
        </header>

        <div style={{ background: "#EFE6DC", borderLeft: `3px solid ${band.color}` }} className="text-sm px-4 py-3 mb-6">
          <span style={{ fontFamily: "ui-sans-serif, system-ui" }}>
            Portfolio demo on a public research dataset. The same interpretable-model pattern
            works for any subscription business — SaaS, telecom, fintech, streaming.
          </span>
        </div>

        <div className="grid md:grid-cols-5 gap-6">
          {/* Controls */}
          <div style={{ background: PANEL, border: `1px solid ${LINE}` }} className="md:col-span-3 p-5">
            <h2 style={{ fontFamily: "ui-sans-serif, system-ui", color: INK_SOFT }} className="text-xs uppercase tracking-wide mb-4">
              Customer profile
            </h2>
            <div className="space-y-5">
              <Field label={`Tenure — ${customer.tenure} months`} help="How long they've been a customer">
                <input type="range" min={0} max={72} step={1} value={customer.tenure}
                  onChange={(e) => update("tenure", Number(e.target.value))}
                  style={{ accentColor: ACCENT, width: "100%" }} />
              </Field>

              <Field label={`Monthly charges — $${customer.MonthlyCharges.toFixed(2)}`}>
                <input type="range" min={18} max={120} step={0.5} value={customer.MonthlyCharges}
                  onChange={(e) => update("MonthlyCharges", Number(e.target.value))}
                  style={{ accentColor: ACCENT, width: "100%" }} />
              </Field>

              <Field label={`Total charges to date — $${customer.TotalCharges.toFixed(0)}`}>
                <input type="range" min={0} max={8500} step={25} value={customer.TotalCharges}
                  onChange={(e) => update("TotalCharges", Number(e.target.value))}
                  style={{ accentColor: ACCENT, width: "100%" }} />
              </Field>

              <div className="grid grid-cols-2 gap-4">
                <Field label="Contract type">
                  <select style={selectStyle} value={customer.Contract} onChange={(e) => update("Contract", e.target.value)}>
                    {CONTRACT_OPTIONS.map((o) => <option key={o} value={o}>{o}</option>)}
                  </select>
                </Field>
                <Field label="Internet service">
                  <select style={selectStyle} value={customer.InternetService} onChange={(e) => update("InternetService", e.target.value)}>
                    {INTERNET_OPTIONS.map((o) => <option key={o} value={o}>{o}</option>)}
                  </select>
                </Field>
              </div>

              <Field label="Payment method">
                <select style={selectStyle} value={customer.PaymentMethod} onChange={(e) => update("PaymentMethod", e.target.value)}>
                  {PAYMENT_OPTIONS.map((o) => <option key={o} value={o}>{o}</option>)}
                </select>
              </Field>

              <div className="flex gap-6 pt-1" style={{ fontFamily: "ui-sans-serif, system-ui" }}>
                <label className="text-sm flex items-center gap-2">
                  <input type="checkbox" checked={customer.PaperlessBilling} onChange={(e) => update("PaperlessBilling", e.target.checked)} />
                  Paperless billing
                </label>
                <label className="text-sm flex items-center gap-2">
                  <input type="checkbox" checked={customer.SeniorCitizen} onChange={(e) => update("SeniorCitizen", e.target.checked)} />
                  Senior citizen
                </label>
              </div>
            </div>
          </div>

          {/* Result */}
          <div className="md:col-span-2 flex flex-col gap-4">
            <div style={{ background: PANEL, border: `1px solid ${LINE}` }} className="p-5">
              <h2 style={{ fontFamily: "ui-sans-serif, system-ui", color: INK_SOFT }} className="text-xs uppercase tracking-wide mb-3">
                Model output
              </h2>
              <div className="flex items-end justify-between mb-2">
                <span style={{ fontFamily: "ui-monospace, monospace", fontSize: "2.25rem", color: band.color, lineHeight: 1 }}>
                  {(pChurn * 100).toFixed(1)}%
                </span>
                <span style={{ fontFamily: "ui-sans-serif, system-ui", color: band.color }} className="text-sm mb-1">
                  {band.label} churn risk
                </span>
              </div>
              <div style={{ background: LINE, height: 6, borderRadius: 3, overflow: "hidden" }}>
                <div style={{ width: `${pChurn * 100}%`, background: band.color, height: "100%" }} />
              </div>
              <p style={{ color: INK_SOFT, fontFamily: "ui-sans-serif, system-ui" }} className="text-xs mt-2">
                predicted probability this customer churns
              </p>
            </div>

            <div style={{ background: PANEL, border: `1px solid ${LINE}` }} className="p-5 flex-1">
              <h2 style={{ fontFamily: "ui-sans-serif, system-ui", color: INK_SOFT }} className="text-xs uppercase tracking-wide mb-3">
                Plain-language explanation
              </h2>
              <button
                onClick={handleExplain}
                style={{ background: ACCENT, color: "#fff", fontFamily: "ui-sans-serif, system-ui" }}
                className="text-sm px-4 py-2 mb-3 w-full transition-colors"
              >
                Explain this result
              </button>
              {explanation && (
                <p style={{ fontFamily: "ui-sans-serif, system-ui", color: INK, lineHeight: 1.6 }} className="text-sm whitespace-pre-wrap">
                  {explanation}
                </p>
              )}
              {!explanation && (
                <p style={{ fontFamily: "ui-sans-serif, system-ui", color: INK_SOFT }} className="text-sm">
                  Adjust the profile, then click above to have the model's reasoning translated
                  into plain language.
                </p>
              )}
            </div>
          </div>
        </div>

        <footer style={{ borderTop: `1px solid ${LINE}`, fontFamily: "ui-sans-serif, system-ui", color: INK_SOFT }} className="mt-8 pt-4 text-xs flex flex-wrap gap-x-6 gap-y-1">
          <span>Accuracy: {(MODEL.metrics.accuracy * 100).toFixed(1)}%</span>
          <span>Recall (churn): {(MODEL.metrics.recallChurn * 100).toFixed(1)}%</span>
          <span>Precision (churn): {(MODEL.metrics.precisionChurn * 100).toFixed(1)}%</span>
          <span>Test set: {MODEL.metrics.testSetSize} customers</span>
        </footer>
      </div>
    </div>
  );
}
