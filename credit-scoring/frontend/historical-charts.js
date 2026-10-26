const chartColors = {
  no: "#4263eb",
  yes: "#f08c45"
};

const chartLayout = {
  autosize: true,
  margin: { l: 55, r: 20, t: 20, b: 65 },
  paper_bgcolor: "white",
  plot_bgcolor: "white",
  font: { family: "Arial, sans-serif", color: "#172033" },
  legend: { orientation: "h", y: -0.25 }
};

const chartConfig = { responsive: true, displaylogo: false };

function histogramTraces(profiles, labels, title) {
  return profiles.map((profile, index) => ({
    type: "bar",
    name: profile.label,
    x: labels,
    y: profile.percentages,
    marker: {
      color: index === 0 ? chartColors.no : chartColors.yes
    },
    hovertemplate: "%{x}<br>%{y:.2f}% of this outcome group<extra></extra>"
  }));
}

async function renderHistoricalCharts() {
  const response = await fetch("/explainability-data.json");

  if (!response.ok) {
    throw new Error("Could not load aggregated historical chart data.");
  }

  const data = await response.json();

  Plotly.newPlot(
    "ageDistributionChart",
    histogramTraces(data.age.profiles, data.age.labels),
    {
      ...chartLayout,
      barmode: "group",
      xaxis: { title: "Age group (years)", tickangle: -35 },
      yaxis: { title: "Share within outcome group (%)", rangemode: "tozero" }
    },
    chartConfig
  );

  Plotly.newPlot(
    "ageRateChart",
    [{
      type: "bar",
      x: data.age.rates.labels,
      y: data.age.rates.percentages,
      marker: { color: chartColors.yes },
      customdata: data.age.rates.counts,
      hovertemplate: "Age group: %{x}<br>Payment difficulties: %{y:.2f}%<br>Applicants: %{customdata:,}<extra></extra>"
    }],
    {
      ...chartLayout,
      showlegend: false,
      xaxis: { title: "Age group (years)" },
      yaxis: { title: "Applicants with TARGET = 1 (%)", rangemode: "tozero" }
    },
    chartConfig
  );

  Plotly.newPlot(
    "incomeDistributionChart",
    histogramTraces(data.income.profiles, data.income.labels),
    {
      ...chartLayout,
      barmode: "group",
      xaxis: { title: "Reported income (thousands)", tickangle: -35 },
      yaxis: { title: "Share within outcome group (%)", rangemode: "tozero" },
      annotations: [{
        text: `Top 1% excluded; cutoff: ${Number(data.income.cutoff).toLocaleString()}`,
        xref: "paper", yref: "paper", x: 0, y: 1.12,
        showarrow: false, xanchor: "left"
      }]
    },
    chartConfig
  );

  const ratio = data.credit_income_ratio.summary;
  const boxTraces = [
    {
      type: "box",
      name: "No recorded difficulties",
      q1: [ratio.no_difficulties.q1],
      median: [ratio.no_difficulties.median],
      q3: [ratio.no_difficulties.q3],
      lowerfence: [ratio.no_difficulties.min],
      upperfence: [ratio.no_difficulties.max],
      marker: { color: chartColors.no },
      boxpoints: false,
      hovertemplate: "Median: %{median:.2f}<br>Q1: %{q1:.2f}<br>Q3: %{q3:.2f}<extra></extra>"
    },
    {
      type: "box",
      name: "Recorded difficulties",
      q1: [ratio.difficulties.q1],
      median: [ratio.difficulties.median],
      q3: [ratio.difficulties.q3],
      lowerfence: [ratio.difficulties.min],
      upperfence: [ratio.difficulties.max],
      marker: { color: chartColors.yes },
      boxpoints: false,
      hovertemplate: "Median: %{median:.2f}<br>Q1: %{q1:.2f}<br>Q3: %{q3:.2f}<extra></extra>"
    }
  ];

  Plotly.newPlot(
    "creditIncomeChart",
    boxTraces,
    {
      ...chartLayout,
      yaxis: { title: "Credit-to-income ratio", range: [0, 10] },
      xaxis: { title: "Historical outcome group" },
      annotations: [{
        text: `Invalid ratios and ratios above 10 omitted`,
        xref: "paper", yref: "paper", x: 0, y: 1.12,
        showarrow: false, xanchor: "left"
      }]
    },
    chartConfig
  );
}

renderHistoricalCharts().catch(error => {
  for (const id of [
    "ageDistributionChart",
    "ageRateChart",
    "incomeDistributionChart",
    "creditIncomeChart"
  ]) {
    const element = document.getElementById(id);
    if (element) element.innerHTML = `<p class="error-message">${error.message}</p>`;
  }
});
