const canvas = document.getElementById("importanceChart");
const message = document.getElementById("importanceMessage");
const refreshButton = document.getElementById("refreshImportance");
let chart;

async function loadImportance() {
  refreshButton.disabled = true;
  message.textContent = "Loading feature importance…";

  try {
    const response = await fetch("/api/explainability");

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Could not load feature importance.");
    }

    const result = await response.json();

    if (!result.features || result.features.length === 0) {
      throw new Error("The model did not return feature importance.");
    }

    message.textContent = `Showing the top ${result.features.length} features by split-based importance.`;

    if (chart) chart.destroy();

    chart = new Chart(canvas, {
      type: "bar",
      data: {
        labels: result.features.map(item => item.feature),
        datasets: [{
          label: "Split importance",
          data: result.features.map(item => item.importance),
          backgroundColor: "#4263eb",
          borderRadius: 4
        }]
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        scales: { x: { beginAtZero: true } },
        plugins: { legend: { display: false } }
      }
    });
  } catch (error) {
    message.textContent = error.message;
  } finally {
    refreshButton.disabled = false;
  }
}

refreshButton.addEventListener("click", loadImportance);
loadImportance();
