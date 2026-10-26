const targetCanvas = document.getElementById("targetChart");

if (targetCanvas) {
  new Chart(targetCanvas, {
    type: "doughnut",
    data: {
      labels: ["No recorded payment difficulties", "Payment difficulties"],
      datasets: [{
        data: [282686, 24825],
        backgroundColor: ["#4263eb", "#f08c45"],
        borderWidth: 0
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: "bottom" }
      }
    }
  });
}

const performanceCanvas = document.getElementById("performanceChart");

if (performanceCanvas) {
  new Chart(performanceCanvas, {
    type: "bar",
    data: {
      labels: ["ROC-AUC", "Average precision"],
      datasets: [{
        label: "Validation score",
        data: [0.7513, 0.2388],
        backgroundColor: ["#4263eb", "#f08c45"],
        borderRadius: 6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: true,
          max: 1,
          ticks: { stepSize: 0.2 }
        }
      },
      plugins: { legend: { display: false } }
    }
  });
}
