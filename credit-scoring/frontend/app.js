const form = document.getElementById("assessmentForm");
const resultPanel = document.getElementById("resultPanel");
const assessButton = document.getElementById("assessButton");
const customerSearch = document.getElementById("customerSearch");
const customerSuggestions = document.getElementById("customerSuggestions");
const customerCount = document.getElementById("customerCount");

let searchTimer;
let requestController;

async function loadCustomers(search = "") {
  if (requestController) {
    requestController.abort();
  }

  requestController = new AbortController();

  try {
    const params = new URLSearchParams({
      search,
      limit: "20",
      offset: "0"
    });

    const response = await fetch(`/api/customers?${params}`, {
      signal: requestController.signal
    });

    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.detail || "Could not load customer suggestions.");
    }

    if (!Array.isArray(result.customer_ids)) {
      throw new Error("Unexpected response from the customer API.");
    }

    customerSuggestions.replaceChildren();

    result.customer_ids.forEach((id) => {
      const option = document.createElement("option");
      option.value = String(id);
      customerSuggestions.appendChild(option);
    });

    customerCount.textContent =
      `Showing ${result.customer_ids.length} suggestions out of ` +
      `${result.total.toLocaleString()} matching customers.`;

  } catch (error) {
    if (error.name === "AbortError") return;

    customerCount.textContent = error.message;
  }
}

customerSearch.addEventListener("input", () => {
  clearTimeout(searchTimer);

  const search = customerSearch.value.trim();

  searchTimer = setTimeout(() => {
    loadCustomers(search);
  }, 250);
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const enteredId = customerSearch.value.trim();

  if (!/^\d+$/.test(enteredId)) {
    customerCount.textContent = "Enter a valid numeric customer ID.";
    customerSearch.focus();
    return;
  }

  const customerId = Number(enteredId);

  if (!Number.isSafeInteger(customerId)) {
    customerCount.textContent = "Customer ID must be a valid whole number.";
    return;
  }

  assessButton.disabled = true;
  assessButton.textContent = "Assessing...";
  resultPanel.innerHTML = '<p class="muted">Calculating risk estimate...</p>';

  try {
    const response = await fetch("/api/assess", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ customer_id: customerId })
    });

    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.detail || "Assessment failed.");
    }

    resultPanel.innerHTML = `
      <p class="eyebrow">ASSESSMENT RESULT</p>
      <h2>Customer ${escapeHtml(result.customer_id)}</h2>
      <p class="muted">Estimated probability of payment difficulties</p>
      <div class="risk-score">${escapeHtml(result.risk_percentage)}%</div>
      <span class="risk-badge ${escapeHtml(String(result.risk_level).toLowerCase())}">
        ${escapeHtml(result.risk_level)} risk
      </span>
      <div class="result-detail">
        <span>Predicted class</span>
        <strong>${escapeHtml(result.predicted_class)}</strong>
      </div>
      <div class="result-detail">
        <span>Model</span>
        <strong>LightGBM</strong>
      </div>
      <p class="disclaimer">${escapeHtml(result.disclaimer)}</p>
      <a class="primary-link" href="/explainability.html">Explore model explainability →</a>
    `;
  } catch (error) {
    resultPanel.innerHTML = `
      <div class="empty-state">
        <h2>Assessment unavailable</h2>
        <p class="error-message">${escapeHtml(error.message)}</p>
      </div>
    `;
  } finally {
    assessButton.disabled = false;
    assessButton.textContent = "Run assessment";
  }
});

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (character) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;"
  })[character]);
}

loadCustomers();
