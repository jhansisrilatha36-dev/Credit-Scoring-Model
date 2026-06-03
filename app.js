// -------------------------------------------------------------
// CreditWise AI Frontend Logic - Random Forest Evaluator & Inference
// -------------------------------------------------------------

// Application State
let appState = {
    dataSummary: null,
    modelMetrics: null,
    isTrained: false,
    activeTab: 'dashboard',
    charts: {} // Store Chart.js objects to manage instances
};

// Global variables to track last results for file downloads
let lastResultA = null;
let lastResultB = null;
let lastInputA = null;
let lastInputB = null;

// Elements
const dom = {
    navItems: document.querySelectorAll('.nav-item'),
    tabContents: document.querySelectorAll('.tab-content'),
    pageTitle: document.getElementById('page-title'),
    pageDescription: document.getElementById('page-description'),
    btnRetrain: document.getElementById('btn-retrain-models'),
    btnTrainAction: document.getElementById('btn-train-action'),
    globalLoader: document.getElementById('global-loader'),
    loaderText: document.getElementById('loader-text'),
    
    // Sidebar
    sidebarTotalRecords: document.getElementById('sidebar-total-records'),
    sidebarDefaultRate: document.getElementById('sidebar-default-rate'),
    
    // Overview Tab
    statTotal: document.getElementById('stat-total-profiles'),
    statCreditworthy: document.getElementById('stat-creditworthy'),
    statDefaults: document.getElementById('stat-defaults'),
    statRiskRate: document.getElementById('stat-risk-rate'),
    
    // Dynamic config elements
    datasetConfigForm: document.getElementById('dataset-config-form'),
    configSize: document.getElementById('config-size'),
    configNoise: document.getElementById('config-noise'),
    sizeLblVal: document.getElementById('size-lbl-val'),
    noiseLblVal: document.getElementById('noise-lbl-val'),
    
    // Benchmark Tab
    trainingAlert: document.getElementById('training-alert'),
    modelResults: document.getElementById('model-results'),
    modelsMetricsCards: document.getElementById('models-metrics-cards'),
    confusionMatrixGrid: document.getElementById('confusion-matrix-grid'),
    
    // Calculator Tab
    calcTrainingAlert: document.getElementById('calc-training-alert'),
    calcLayout: document.getElementById('calc-layout'),
    creditForm: document.getElementById('credit-form'),
    resultPlaceholder: document.getElementById('result-placeholder'),
    resultContent: document.getElementById('result-actual-content'),
    resultCompareContent: document.getElementById('result-compare-content'),
    toggleCompareMode: document.getElementById('toggle-compare-mode'),
    formColB: document.getElementById('form-col-b'),
    formsSplitLayout: document.querySelector('.forms-split-layout'),
    btnExportReport: document.getElementById('btn-export-report'),
    btnExportReportCompare: document.getElementById('btn-export-report-compare'),
    
    // Calculator Single Results
    decisionStatus: document.getElementById('decision-status'),
    decisionBadge: document.getElementById('decision-badge'),
    decisionIcon: document.getElementById('decision-icon'),
    decisionStatement: document.getElementById('decision-statement'),
    decisionSubtext: document.getElementById('decision-subtext'),
    overallProbText: document.getElementById('overall-probability-text'),
    overallProbBar: document.getElementById('overall-probability-bar'),
    overallRiskLevel: document.getElementById('overall-risk-level'),
    modelsPredictionsList: document.getElementById('models-predictions-list'),
    
    // Calculator Compare Results
    decisionStatusA: document.getElementById('decision-status-a'),
    decisionBadgeA: document.getElementById('decision-badge-a'),
    decisionIconA: document.getElementById('decision-icon-a'),
    decisionStatementA: document.getElementById('decision-statement-a'),
    decisionSubtextA: document.getElementById('decision-subtext-a'),
    overallProbTextA: document.getElementById('overall-probability-text-a'),
    overallProbBarA: document.getElementById('overall-probability-bar-a'),
    modelsPredictionsListA: document.getElementById('models-predictions-list-a'),
    
    decisionStatusB: document.getElementById('decision-status-b'),
    decisionBadgeB: document.getElementById('decision-badge-b'),
    decisionIconB: document.getElementById('decision-icon-b'),
    decisionStatementB: document.getElementById('decision-statement-b'),
    decisionSubtextB: document.getElementById('decision-subtext-b'),
    overallProbTextB: document.getElementById('overall-probability-text-b'),
    overallProbBarB: document.getElementById('overall-probability-bar-b'),
    modelsPredictionsListB: document.getElementById('models-predictions-list-b')
};

// Global Chart Options helper
const getChartConfig = (type, labels, datasets, options = {}) => {
    return {
        type: type,
        data: { labels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: '#475569', font: { family: 'Inter', size: 11 } }
                },
                tooltip: {
                    padding: 10,
                    bodyFont: { family: 'Inter' },
                    titleFont: { family: 'Outfit', weight: 'bold' }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(0, 0, 0, 0.05)' },
                    ticks: { color: '#475569', font: { family: 'Inter', size: 10 } }
                },
                y: {
                    grid: { color: 'rgba(0, 0, 0, 0.05)' },
                    ticks: { color: '#475569', font: { family: 'Inter', size: 10 } }
                }
            },
            ...options
        }
    };
};

// Loader Handlers
function showLoader(text = 'Processing...') {
    dom.loaderText.textContent = text;
    dom.globalLoader.classList.remove('hidden');
}

function hideLoader() {
    dom.globalLoader.classList.add('hidden');
}


async function initApp() {
    showLoader('Initializing credit profiles...');
    try {
        const response = await fetch('/api/data-summary');
        if (!response.ok) throw new Error('Failed to load dataset details.');
        const data = await response.json();
        
        appState.dataSummary = data;
        
        // Update dashboard UI elements
        updateDashboardUI(data);
        // Render EDA distribution charts
        renderEDACharts(data.charts);
        
    } catch (error) {
        console.error(error);
        alert('Error connecting to backend API. Please make sure the server is running.');
    } finally {
        hideLoader();
    }
}

function updateDashboardUI(data) {
    // Sidebar
    dom.sidebarTotalRecords.textContent = data.total_records.toLocaleString();
    dom.sidebarDefaultRate.textContent = data.default_rate + '%';
    
    // Overview tab stats
    dom.statTotal.textContent = data.total_records.toLocaleString();
    dom.statCreditworthy.textContent = data.non_defaults.toLocaleString();
    dom.statDefaults.textContent = data.defaults.toLocaleString();
    dom.statRiskRate.textContent = data.default_rate + '%';
}


function renderEDACharts(chartData) {
    // 1. Credit Score Range Distribution vs Default
    if (appState.charts.creditScore) appState.charts.creditScore.destroy();
    const csCtx = document.getElementById('chart-credit-score').getContext('2d');
    appState.charts.creditScore = new Chart(csCtx, getChartConfig('bar', chartData.credit_score_dist.labels, [
        {
            label: 'Creditworthy (Non-Default)',
            data: chartData.credit_score_dist.non_defaults,
            backgroundColor: '#10b981',
            borderRadius: 4
        },
        {
            label: 'High Risk (Default)',
            data: chartData.credit_score_dist.defaults,
            backgroundColor: '#ef4444',
            borderRadius: 4
        }
    ], {
        scales: {
            x: { stacked: true, grid: { display: false } },
            y: { stacked: true, grid: { color: 'rgba(0, 0, 0, 0.05)' } }
        }
    }));

    // 2. DTI vs Default Distribution
    if (appState.charts.dti) appState.charts.dti.destroy();
    const dtiCtx = document.getElementById('chart-dti').getContext('2d');
    appState.charts.dti = new Chart(dtiCtx, getChartConfig('bar', chartData.dti_dist.labels, [
        {
            label: 'Creditworthy',
            data: chartData.dti_dist.non_defaults,
            backgroundColor: '#10b981',
            borderRadius: 4
        },
        {
            label: 'Default',
            data: chartData.dti_dist.defaults,
            backgroundColor: '#ef4444',
            borderRadius: 4
        }
    ], {
        scales: {
            x: { stacked: true, grid: { display: false } },
            y: { stacked: true, grid: { color: 'rgba(0, 0, 0, 0.05)' } }
        }
    }));

    // 3. Home Ownership vs Default
    if (appState.charts.home) appState.charts.home.destroy();
    const homeCtx = document.getElementById('chart-home').getContext('2d');
    appState.charts.home = new Chart(homeCtx, getChartConfig('bar', chartData.home_ownership.labels, [
        {
            label: 'Creditworthy',
            data: chartData.home_ownership.non_defaults,
            backgroundColor: '#4f46e5',
            borderRadius: 4
        },
        {
            label: 'Default',
            data: chartData.home_ownership.defaults,
            backgroundColor: '#d97706',
            borderRadius: 4
        }
    ]));
}


async function trainModels() {
    showLoader('Fitting and optimizing Random Forest Classifier...');
    try {
        const response = await fetch('/api/train', { method: 'POST' });
        if (!response.ok) throw new Error('Training request failed.');
        const metrics = await response.json();
        
        appState.modelMetrics = metrics;
        appState.isTrained = true;
        
        // Hide initial placeholders, show results
        dom.trainingAlert.classList.add('hidden');
        dom.modelResults.classList.remove('hidden');
        dom.calcTrainingAlert.classList.add('hidden');
        dom.calcLayout.classList.remove('hidden');
        
        // Render results components
        renderModelMetricsCards(metrics);
        renderROCCurve(metrics);
        renderFeatureImportance();
        renderConfusionMatrices(metrics);
        
    } catch (error) {
        console.error(error);
        alert('An error occurred during model training.');
    } finally {
        hideLoader();
    }
}

// Render model scorecard grid
function renderModelMetricsCards(metrics) {
    dom.modelsMetricsCards.innerHTML = '';
    
    const details = metrics['Random Forest'];
    if (!details) return;
    
    const m = details.metrics;
    const cardHtml = `
        <div class="model-metric-card" style="max-width: 500px; margin: 0 auto;">
            <div class="card-top">
                <span class="model-name">Random Forest Scorecard</span>
                <span class="model-badge" style="background-color: rgba(79, 70, 229, 0.08); color: var(--color-primary); border-color: rgba(79, 70, 229, 0.15);">Optimal Fit</span>
            </div>
            <div class="metrics-list">
                <div class="metric-item">
                    <span class="metric-name">Accuracy (Overall Correctness)</span>
                    <span class="metric-val">${(m.Accuracy * 100).toFixed(1)}%</span>
                </div>
                <div class="metric-item">
                    <span class="metric-name">Precision (Low False Positive rate)</span>
                    <span class="metric-val">${(m.Precision * 100).toFixed(1)}%</span>
                </div>
                <div class="metric-item">
                    <span class="metric-name">Recall (Capture rate of defaults)</span>
                    <span class="metric-val">${(m.Recall * 100).toFixed(1)}%</span>
                </div>
                <div class="metric-item">
                    <span class="metric-name">F1-Score (Harmonic Mean)</span>
                    <span class="metric-val">${m['F1-Score'].toFixed(3)}</span>
                </div>
                <div class="metric-item" style="border-top: 1px solid var(--border-color); padding-top: 12px; margin-top: 6px;">
                    <span class="metric-name" style="font-weight: 600; color: var(--text-primary);">ROC-AUC (Model Discriminative Power)</span>
                    <span class="metric-val" style="color: var(--color-primary); font-size: 16px; font-weight: 700;">${m['ROC-AUC'].toFixed(3)}</span>
                </div>
            </div>
        </div>
    `;
    dom.modelsMetricsCards.insertAdjacentHTML('beforeend', cardHtml);
}

// Render ROC curves side by side
function renderROCCurve(metrics) {
    if (appState.charts.roc) appState.charts.roc.destroy();
    
    const details = metrics['Random Forest'];
    if (!details) return;
    
    const datasets = [];
    
    // Baseline 50-50 line
    datasets.push({
        label: 'Random Baseline (AUC = 0.500)',
        data: [{fpr: 0, tpr: 0}, {fpr: 1, tpr: 1}],
        borderColor: 'rgba(0, 0, 0, 0.15)',
        borderDash: [5, 5],
        borderWidth: 1.5,
        fill: false,
        pointStyle: 'none',
        pointRadius: 0,
        parsing: { xAxisKey: 'fpr', yAxisKey: 'tpr' }
    });
    
    datasets.push({
        label: `Random Forest Classifier (AUC = ${details.metrics['ROC-AUC'].toFixed(3)})`,
        data: details.roc_curve,
        borderColor: '#4f46e5',
        borderWidth: 2.5,
        fill: false,
        pointRadius: 1.5,
        tension: 0.1,
        parsing: { xAxisKey: 'fpr', yAxisKey: 'tpr' }
    });
    
    const rocCtx = document.getElementById('chart-roc').getContext('2d');
    appState.charts.roc = new Chart(rocCtx, {
        type: 'line',
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: '#475569', font: { family: 'Inter', size: 10 } } },
                tooltip: { mode: 'index', intersect: false }
            },
            scales: {
                x: {
                    type: 'linear',
                    title: { display: true, text: 'False Positive Rate (FPR)', color: '#475569' },
                    grid: { color: 'rgba(0, 0, 0, 0.05)' },
                    ticks: { color: '#475569' },
                    min: 0, max: 1
                },
                y: {
                    type: 'linear',
                    title: { display: true, text: 'True Positive Rate (TPR)', color: '#475569' },
                    grid: { color: 'rgba(0, 0, 0, 0.05)' },
                    ticks: { color: '#475569' },
                    min: 0, max: 1
                }
            }
        }
    });
}

// Render feature weights / importance
function renderFeatureImportance() {
    if (!appState.modelMetrics) return;
    
    const details = appState.modelMetrics['Random Forest'];
    if (!details) return;
    
    const importanceData = details.feature_importance;
    const labels = importanceData.map(item => {
        return item.feature.replace('num__', '').replace('cat__', '').replace('_', ': ');
    });
    const values = importanceData.map(item => item.value);
    
    if (appState.charts.importance) appState.charts.importance.destroy();
    
    const impCtx = document.getElementById('chart-importance').getContext('2d');
    appState.charts.importance = new Chart(impCtx, getChartConfig('bar', labels, [
        {
            label: 'Gini Importance (MDI)',
            data: values,
            backgroundColor: 'rgba(79, 70, 229, 0.75)',
            borderColor: '#4f46e5',
            borderWidth: 1,
            borderRadius: 3
        }
    ], {
        indexAxis: 'y',
        plugins: {
            legend: { display: false }
        }
    }));
}

// Render Confusion Matrix
function renderConfusionMatrices(metrics) {
    dom.confusionMatrixGrid.innerHTML = '';
    
    const details = metrics['Random Forest'];
    if (!details) return;
    
    const cm = details.confusion_matrix;
    const total = cm.TN + cm.FP + cm.FN + cm.TP;
    const getPct = (val) => ((val / total) * 100).toFixed(1) + '%';
    
    const matrixHtml = `
        <div class="cm-card" style="max-width: 400px; margin: 0 auto;">
            <h4 class="cm-title">Random Forest Classifier</h4>
            <div class="cm-grid-2x2">
                <div class="cm-header-lbl"></div>
                <div class="cm-header-lbl">Pred Low</div>
                <div class="cm-header-lbl">Pred High</div>
                
                <div class="cm-header-lbl">Act Low</div>
                <div class="cm-cell tn">
                    <strong>${cm.TN}</strong>
                    <span>TN (${getPct(cm.TN)})</span>
                </div>
                <div class="cm-cell fp">
                    <strong>${cm.FP}</strong>
                    <span>FP (${getPct(cm.FP)})</span>
                </div>
                
                <div class="cm-header-lbl">Act High</div>
                <div class="cm-cell fn">
                    <strong>${cm.FN}</strong>
                    <span>FN (${getPct(cm.FN)})</span>
                </div>
                <div class="cm-cell tp">
                    <strong>${cm.TP}</strong>
                    <span>TP (${getPct(cm.TP)})</span>
                </div>
            </div>
        </div>
    `;
    dom.confusionMatrixGrid.insertAdjacentHTML('beforeend', matrixHtml);
}


async function runPrediction(e) {
    e.preventDefault();
    if (!appState.isTrained) {
        alert('Model must be trained before predictions can run.');
        return;
    }
    
    const isCompare = dom.toggleCompareMode.checked;
    showLoader(isCompare ? 'Comparing credit profiles...' : 'Calculating credit risk...');
    
    const formData = new FormData(dom.creditForm);
    const payloadA = {};
    const payloadB = {};
    
    formData.forEach((value, key) => {
        if (key.endsWith('_B')) {
            const cleanKey = key.replace('_B', '');
            payloadB[cleanKey] = value;
        } else {
            payloadA[key] = value;
        }
    });
    
    try {
        if (isCompare) {
            // Predict profile A
            const resA = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payloadA)
            });
            if (!resA.ok) throw new Error('Applicant A calculation failed.');
            const resultA = await resA.json();
            
            // Predict profile B
            const resB = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payloadB)
            });
            if (!resB.ok) throw new Error('Applicant B calculation failed.');
            const resultB = await resB.json();
            
            // Render Comparative columns
            dom.resultPlaceholder.classList.add('hidden');
            dom.resultContent.classList.add('hidden');
            dom.resultCompareContent.classList.remove('hidden');
            
            displayCompareResults(resultA, resultB, payloadA, payloadB);
            
        } else {
            // Predict single profile
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payloadA)
            });
            if (!response.ok) throw new Error('Prediction API call failed.');
            const result = await response.json();
            
            // Render single results
            dom.resultPlaceholder.classList.add('hidden');
            dom.resultCompareContent.classList.add('hidden');
            dom.resultContent.classList.remove('hidden');
            
            displaySingleResults(result, payloadA);
        }
    } catch (error) {
        console.error(error);
        alert('An error occurred: ' + error.message);
    } finally {
        hideLoader();
    }
}

// Display single prediction scorecard
function displaySingleResults(result, inputs) {
    lastResultA = result;
    lastInputA = inputs;
    lastResultB = null;
    lastInputB = null;
    
    const rfPred = result.predictions['Random Forest'];
    if (!rfPred) return;
    
    const probability = rfPred.probability;
    const probabilityPercentage = (probability * 100).toFixed(1);
    
    dom.overallProbText.textContent = probabilityPercentage + '%';
    dom.overallProbBar.style.width = probabilityPercentage + '%';
    
    dom.overallProbBar.className = 'gauge-bar-inner';
    dom.decisionStatus.className = 'decision-status-box';
    
    let decision = "Approved";
    let riskLabel = "Low Risk";
    let statement = "Applicant demonstrates high creditworthiness.";
    let subtext = "Low default probability detected by Random Forest model.";
    
    if (probability < 0.20) {
        dom.overallProbBar.classList.add('success');
        dom.decisionStatus.classList.add('success');
        dom.decisionIcon.className = 'fa-solid fa-circle-check';
        dom.decisionBadge.textContent = 'Approved (A)';
        riskLabel = "Low Risk";
    } else if (probability < 0.45) {
        dom.overallProbBar.classList.add('warning');
        dom.decisionStatus.classList.add('success'); 
        dom.decisionIcon.className = 'fa-solid fa-circle-exclamation';
        dom.decisionBadge.textContent = 'Conditional (B)';
        statement = "Applicant demonstrates moderate risk profiles.";
        subtext = "Requires secondary review based on criteria thresholds.";
        riskLabel = "Medium Risk";
    } else {
        dom.overallProbBar.classList.add('danger');
        dom.decisionStatus.classList.add('danger');
        dom.decisionIcon.className = 'fa-solid fa-circle-xmark';
        dom.decisionBadge.textContent = 'Declined';
        decision = "Declined";
        statement = "High risk of default detected.";
        subtext = "Applicant defaults baseline risk thresholds.";
        riskLabel = "High Risk";
    }
    
    dom.overallRiskLevel.textContent = `Risk Level: ${riskLabel}`;
    dom.decisionStatement.textContent = statement;
    dom.decisionSubtext.textContent = subtext;
    
    // Render single model indicator details
    dom.modelsPredictionsList.innerHTML = `
        <div class="model-prob-item">
            <span class="model-lbl">Random Forest Classifier:</span>
            <span class="model-prob-val">${probabilityPercentage}% probability (${decision === 'Approved' || decision === 'Conditional' ? 'Approved' : 'Declined'})</span>
        </div>
    `;
    
    // Feature explanations bar chart
    renderLocalExplanations(result.explanations);
}

// Display side-by-side comparison scorecards
function displayCompareResults(resultA, resultB, inputsA, inputsB) {
    lastResultA = resultA;
    lastResultB = resultB;
    lastInputA = inputsA;
    lastInputB = inputsB;
    
    const colors = { success: 'success', warning: 'warning', danger: 'danger' };
    
    // Applicant A Score
    const rfPredA = resultA.predictions['Random Forest'];
    if (!rfPredA) return;
    const probA = rfPredA.probability;
    const probAPct = (probA * 100).toFixed(1);
    
    dom.overallProbabilityTextA.textContent = probAPct + '%';
    dom.overallProbBarA.style.width = probAPct + '%';
    dom.overallProbBarA.className = 'gauge-bar-inner';
    dom.decisionStatusA.className = 'decision-status-box';
    
    let decisionA = "Approved";
    let subA = "Low default risk.";
    
    if (probA < 0.20) {
        dom.overallProbBarA.classList.add(colors.success);
        dom.decisionStatusA.classList.add('success');
        dom.decisionIconA.className = 'fa-solid fa-circle-check';
        dom.decisionBadgeA.textContent = 'Approved (A)';
    } else if (probA < 0.45) {
        dom.overallProbBarA.classList.add(colors.warning);
        dom.decisionStatusA.classList.add('success');
        dom.decisionIconA.className = 'fa-solid fa-circle-exclamation';
        dom.decisionBadgeA.textContent = 'Conditional (B)';
        decisionA = "Conditional";
        subA = "Review collateral details.";
    } else {
        dom.overallProbBarA.classList.add(colors.danger);
        dom.decisionStatusA.classList.add('danger');
        dom.decisionIconA.className = 'fa-solid fa-circle-xmark';
        dom.decisionBadgeA.textContent = 'Declined';
        decisionA = "Declined";
        subA = "Exceeds debt thresholds.";
    }
    dom.decisionStatementA.textContent = decisionA;
    dom.decisionSubtextA.textContent = subA;
    
    dom.modelsPredictionsListA.innerHTML = `
        <div class="model-prob-item">
            <span class="model-lbl">Random Forest:</span>
            <span class="model-prob-val">${probAPct}% probability</span>
        </div>
    `;
    
    // Applicant B Score
    const rfPredB = resultB.predictions['Random Forest'];
    if (!rfPredB) return;
    const probB = rfPredB.probability;
    const probBPct = (probB * 100).toFixed(1);
    
    dom.overallProbabilityTextB.textContent = probBPct + '%';
    dom.overallProbBarB.style.width = probBPct + '%';
    dom.overallProbBarB.className = 'gauge-bar-inner';
    dom.decisionStatusB.className = 'decision-status-box';
    
    let decisionB = "Approved";
    let subB = "Low default risk.";
    
    if (probB < 0.20) {
        dom.overallProbBarB.classList.add(colors.success);
        dom.decisionStatusB.classList.add('success');
        dom.decisionIconB.className = 'fa-solid fa-circle-check';
        dom.decisionBadgeB.textContent = 'Approved (A)';
    } else if (probB < 0.45) {
        dom.overallProbBarB.classList.add(colors.warning);
        dom.decisionStatusB.classList.add('success');
        dom.decisionIconB.className = 'fa-solid fa-circle-exclamation';
        dom.decisionBadgeB.textContent = 'Conditional (B)';
        decisionB = "Conditional";
        subB = "Review collateral details.";
    } else {
        dom.overallProbBarB.classList.add(colors.danger);
        dom.decisionStatusB.classList.add('danger');
        dom.decisionIconB.className = 'fa-solid fa-circle-xmark';
        dom.decisionBadgeB.textContent = 'Declined';
        decisionB = "Declined";
        subB = "Exceeds debt thresholds.";
    }
    dom.decisionStatementB.textContent = decisionB;
    dom.decisionSubtextB.textContent = subB;
    
    dom.modelsPredictionsListB.innerHTML = `
        <div class="model-prob-item">
            <span class="model-lbl">Random Forest:</span>
            <span class="model-prob-val">${probBPct}% probability</span>
        </div>
    `;
}

function renderLocalExplanations(explanations) {
    const labels = explanations.map(item => `${item.feature} (${item.actual_value})`);
    const values = explanations.map(item => item.value);
    
    if (appState.charts.explanation) appState.charts.explanation.destroy();
    
    const colors = values.map(v => v >= 0 ? 'rgba(225, 29, 72, 0.75)' : 'rgba(16, 185, 129, 0.75)');
    const borderColors = values.map(v => v >= 0 ? '#e11d48' : '#10b981');
    
    const expCtx = document.getElementById('chart-explanation').getContext('2d');
    appState.charts.explanation = new Chart(expCtx, getChartConfig('bar', labels, [
        {
            label: 'Risk Contribution',
            data: values,
            backgroundColor: colors,
            borderColor: borderColors,
            borderWidth: 1,
            borderRadius: 3
        }
    ], {
        indexAxis: 'y',
        plugins: { legend: { display: false } },
        scales: {
            x: {
                title: { display: true, text: 'Contribution Strength (MDI Weighted Scaling)', color: '#475569', font: { size: 9 } },
                grid: { color: 'rgba(0, 0, 0, 0.05)' }
            }
        }
    }));
}

// -------------------------------------------------------------
// Report Exporters (Downloads)
// -------------------------------------------------------------
function downloadSingleReport() {
    if (!lastResultA || !lastInputA) return;
    
    const rfPred = lastResultA.predictions['Random Forest'];
    if (!rfPred) return;
    
    const prob = (rfPred.probability * 100).toFixed(1);
    const decision = rfPred.probability < 0.20 ? 'APPROVED' : rfPred.probability < 0.45 ? 'CONDITIONAL' : 'DECLINED';
    
    let report = `====================================================\n`;
    report += `       CREDITWISE AI - SINGLE PROFILE RISK REPORT   \n`;
    report += `====================================================\n`;
    report += `Generated On: ${new Date().toLocaleString()}\n`;
    report += `Lending Status: ${decision}\n`;
    report += `Model Score (Default Probability): ${prob}%\n`;
    report += `Classifier: Optimized Random Forest Ensemble (150 estimators)\n\n`;
    
    report += `--- Applicant Data Profile ---\n`;
    Object.entries(lastInputA).forEach(([key, val]) => {
        report += `${key.padEnd(20)}: ${val}\n`;
    });
    
    report += `\n--- Local Risk Contribution Weights (RF Gini Scaled) ---\n`;
    lastResultA.explanations.forEach(item => {
        report += `${item.feature.padEnd(25)} (${item.actual_value.padEnd(8)}) : ${item.value >= 0 ? '+' : ''}${item.value}\n`;
    });
    
    report += `\n====================================================\n`;
    report += `Disclaimer: Automated assessment based strictly on mathematical parameters.\n`;
    
    triggerFileDownload(report, 'creditwise_credit_report.txt');
}

function downloadCompareReport() {
    if (!lastResultA || !lastResultB || !lastInputA || !lastInputB) return;
    
    const rfPredA = lastResultA.predictions['Random Forest'];
    const rfPredB = lastResultB.predictions['Random Forest'];
    if (!rfPredA || !rfPredB) return;
    
    let report = `====================================================\n`;
    report += `       CREDITWISE AI - COMPARATIVE CREDIT RISK REPORT\n`;
    report += `====================================================\n`;
    report += `Generated On: ${new Date().toLocaleString()}\n\n`;
    
    report += `----------------------------------------------------\n`;
    report += `Attribute                 Applicant A        Applicant B\n`;
    report += `----------------------------------------------------\n`;
    const keys = ['Age', 'Income', 'EmploymentYears', 'HomeOwnership', 'CreditScore', 'DebtToIncomeRatio', 'PaymentHistory', 'LoanAmount', 'LoanPurpose'];
    keys.forEach(k => {
        const valA = String(lastInputA[k]).padEnd(18);
        const valB = String(lastInputB[k]);
        report += `${k.padEnd(26)}${valA}${valB}\n`;
    });
    report += `----------------------------------------------------\n`;
    report += `DEFAULT PROBABILITY       ${(rfPredA.probability * 100).toFixed(1)}%`.padEnd(52) + `${(rfPredB.probability * 100).toFixed(1)}%\n`;
    report += `DECISION STATUS           ${rfPredA.probability < 0.20 ? 'APPROVED' : rfPredA.probability < 0.45 ? 'CONDITIONAL' : 'DECLINED'}`.padEnd(52) + `${rfPredB.probability < 0.20 ? 'APPROVED' : rfPredB.probability < 0.45 ? 'CONDITIONAL' : 'DECLINED'}\n`;
    report += `----------------------------------------------------\n\n`;
    
    report += `Assessment model: CreditWise AI optimized Random Forest Classifier.\n`;
    report += `====================================================\n`;
    triggerFileDownload(report, 'creditwise_comparative_report.txt');
}

function triggerFileDownload(text, filename) {
    const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

function setupEventListeners() {
    // 1. Navigation tabs switcher
    dom.navItems.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');
            
            dom.navItems.forEach(n => n.classList.remove('active'));
            btn.classList.add('active');
            
            dom.tabContents.forEach(tc => tc.classList.remove('active'));
            document.getElementById(`tab-${targetTab}`).classList.add('active');
            
            appState.activeTab = targetTab;
            if (targetTab === 'dashboard') {
                dom.pageTitle.textContent = 'CreditWise AI Dashboard';
                dom.pageDescription.textContent = 'Statistical analysis and EDA of credit history records';
            } else if (targetTab === 'benchmarking') {
                dom.pageTitle.textContent = 'Model Evaluation';
                dom.pageDescription.textContent = 'Performance metrics and optimized classification scorecards';
            } else if (targetTab === 'calculator') {
                dom.pageTitle.textContent = 'Individual Credit Risk Calculator';
                dom.pageDescription.textContent = 'Estimate applicant creditworthiness in real-time';
            }
        });
    });

    // 2. Train Action Buttons
    dom.btnRetrain.addEventListener('click', trainModels);
    dom.btnTrainAction.addEventListener('click', trainModels);
    
    // 3. Form Submission (Predictions)
    dom.creditForm.addEventListener('submit', runPrediction);
    
    // 4. Config Form sliders label rendering & submit
    dom.configSize.addEventListener('input', (e) => {
        dom.sizeLblVal.textContent = parseInt(e.target.value).toLocaleString();
    });
    dom.configNoise.addEventListener('input', (e) => {
        dom.noiseLblVal.textContent = Math.round(e.target.value * 100) + '%';
    });
    
    dom.datasetConfigForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        showLoader('Regenerating credit dataset...');
        
        const size = dom.configSize.value;
        const noise = dom.configNoise.value;
        const riskLevel = document.getElementById('config-risk-level').value;
        
        try {
            const res = await fetch('/api/generate-data', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ size, noise, risk_level: riskLevel })
            });
            
            if (!res.ok) throw new Error('Failed to rebuild dataset database.');
            const data = await res.json();
            
            appState.dataSummary = data;
            appState.isTrained = false; // Force retraining
            
            // Reset UI States
            updateDashboardUI(data);
            renderEDACharts(data.charts);
            
            dom.trainingAlert.classList.remove('hidden');
            dom.modelResults.classList.add('hidden');
            dom.calcTrainingAlert.classList.remove('hidden');
            dom.calcLayout.classList.add('hidden');
            
            dom.resultPlaceholder.classList.remove('hidden');
            dom.resultContent.classList.add('hidden');
            dom.resultCompareContent.classList.add('hidden');
            
            alert('Credit dataset reconstructed! Please retrain the model to view scores.');
            
        } catch (error) {
            console.error(error);
            alert('Reconstruction failed: ' + error.message);
        } finally {
            hideLoader();
        }
    });
    
    // 5. Compare mode checkbox toggle
    dom.toggleCompareMode.addEventListener('change', (e) => {
        const compare = e.target.checked;
        if (compare) {
            dom.formColB.classList.remove('hidden');
            dom.formsSplitLayout.classList.add('compare-active');
            
            // Mark fields required
            dom.formColB.querySelectorAll('input, select').forEach(el => {
                el.setAttribute('required', 'true');
            });
        } else {
            dom.formColB.classList.add('hidden');
            dom.formsSplitLayout.classList.remove('compare-active');
            
            // Remove required
            dom.formColB.querySelectorAll('input, select').forEach(el => {
                el.removeAttribute('required');
            });
        }
        
        // Clear old visual selections
        dom.resultPlaceholder.classList.remove('hidden');
        dom.resultContent.classList.add('hidden');
        dom.resultCompareContent.classList.add('hidden');
    });
    
    // 6. Download triggers
    dom.btnExportReport.addEventListener('click', downloadSingleReport);
    dom.btnExportReportCompare.addEventListener('click', downloadCompareReport);
}

// Run application initialization
window.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    initApp();
});
