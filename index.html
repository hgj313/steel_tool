<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>钢材优化系统</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --primary: #3498db;
            --primary-dark: #2980b9;
            --secondary: #2ecc71;
            --warning: #f39c12;
            --danger: #e74c3c;
            --dark: #2c3e50;
            --light: #ecf0f1;
            --gray: #95a5a6;
            --border-radius: 8px;
            --shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            --transition: all 0.3s ease;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f5f7fa;
            color: #333;
            line-height: 1.6;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }

        header {
            background: linear-gradient(135deg, var(--primary), var(--primary-dark));
            color: white;
            padding: 20px;
            border-radius: var(--border-radius);
            margin-bottom: 25px;
            box-shadow: var(--shadow);
            position: relative;
            overflow: hidden;
        }

        header::before {
            content: "";
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0) 70%);
            z-index: 0;
        }

        .header-content {
            position: relative;
            z-index: 1;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        h1 {
            font-size: 2.2rem;
            margin-bottom: 5px;
        }

        .subtitle {
            font-size: 1.1rem;
            opacity: 0.9;
        }

        .theme-switch {
            background: rgba(255, 255, 255, 0.2);
            border: none;
            color: white;
            padding: 8px 15px;
            border-radius: 20px;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 8px;
            transition: var(--transition);
        }

        .theme-switch:hover {
            background: rgba(255, 255, 255, 0.3);
        }

        .dashboard {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 25px;
        }

        @media (max-width: 992px) {
            .dashboard {
                grid-template-columns: 1fr;
            }
        }

        .card {
            background: white;
            border-radius: var(--border-radius);
            box-shadow: var(--shadow);
            padding: 25px;
            transition: var(--transition);
        }

        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
        }

        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 1px solid #eee;
        }

        .card-title {
            font-size: 1.4rem;
            color: var(--dark);
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .card-title i {
            color: var(--primary);
        }

        .card-actions {
            display: flex;
            gap: 10px;
        }

        .btn {
            padding: 8px 16px;
            border: none;
            border-radius: var(--border-radius);
            cursor: pointer;
            font-weight: 500;
            transition: var(--transition);
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }

        .btn-primary {
            background: var(--primary);
            color: white;
        }

        .btn-primary:hover {
            background: var(--primary-dark);
        }

        .btn-secondary {
            background: var(--secondary);
            color: white;
        }

        .btn-secondary:hover {
            background: #27ae60;
        }

        .btn-warning {
            background: var(--warning);
            color: white;
        }

        .btn-warning:hover {
            background: #e67e22;
        }

        .btn-outline {
            background: transparent;
            border: 1px solid var(--gray);
            color: var(--dark);
        }

        .btn-outline:hover {
            background: #f8f9fa;
        }

        .input-group {
            margin-bottom: 15px;
        }

        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 500;
        }

        input[type="text"],
        input[type="number"],
        select {
            width: 100%;
            padding: 12px;
            border: 1px solid #ddd;
            border-radius: var(--border-radius);
            font-size: 1rem;
            transition: var(--transition);
        }

        input:focus, select:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.2);
        }

        .steel-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }

        .steel-table th {
            background-color: #f8f9fa;
            text-align: left;
            padding: 12px 15px;
            font-weight: 600;
            border-bottom: 2px solid #eee;
        }

        .steel-table td {
            padding: 12px 15px;
            border-bottom: 1px solid #eee;
        }

        .steel-table tr:hover td {
            background-color: #f8f9fa;
        }

        .delete-btn {
            color: var(--danger);
            background: none;
            border: none;
            cursor: pointer;
            font-size: 1.1rem;
            transition: var(--transition);
        }

        .delete-btn:hover {
            transform: scale(1.1);
        }

        .progress-section {
            background: white;
            border-radius: var(--border-radius);
            box-shadow: var(--shadow);
            padding: 25px;
            margin-bottom: 25px;
            display: flex;
            flex-direction: column;
            gap: 15px;
        }

        .progress-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .progress-container {
            height: 12px;
            background: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
            margin: 15px 0;
        }

        .progress-bar {
            height: 100%;
            background: linear-gradient(90deg, var(--primary), var(--secondary));
            border-radius: 10px;
            width: 0%;
            transition: width 0.5s ease;
        }

        .progress-info {
            display: flex;
            justify-content: space-between;
            font-size: 0.9rem;
            color: var(--gray);
        }

        .results-section {
            background: white;
            border-radius: var(--border-radius);
            box-shadow: var(--shadow);
            padding: 25px;
            margin-bottom: 25px;
        }

        .summary-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 25px;
        }

        .summary-card {
            background: #f8f9fa;
            border-radius: var(--border-radius);
            padding: 20px;
            text-align: center;
            transition: var(--transition);
        }

        .summary-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.05);
        }

        .summary-card h3 {
            font-size: 1.1rem;
            color: var(--gray);
            margin-bottom: 10px;
        }

        .summary-card .value {
            font-size: 2.2rem;
            font-weight: 700;
            color: var(--primary);
        }

        .summary-card .value.warning {
            color: var(--warning);
        }

        .summary-card .value.success {
            color: var(--secondary);
        }

        .chart-container {
            height: 300px;
            margin-top: 25px;
        }

        footer {
            text-align: center;
            padding: 20px;
            color: var(--gray);
            font-size: 0.9rem;
            border-top: 1px solid #eee;
            margin-top: 20px;
        }

        .notification {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px 20px;
            border-radius: var(--border-radius);
            background: white;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            display: flex;
            align-items: center;
            gap: 10px;
            transform: translateX(150%);
            transition: transform 0.3s ease;
            z-index: 1000;
        }

        .notification.show {
            transform: translateX(0);
        }

        .notification.success {
            border-left: 4px solid var(--secondary);
        }

        .notification.warning {
            border-left: 4px solid var(--warning);
        }

        .notification i {
            font-size: 1.5rem;
        }

        .notification.success i {
            color: var(--secondary);
        }

        .notification.warning i {
            color: var(--warning);
        }

        /* Dark mode styles */
        body.dark-mode {
            background-color: #1a1d21;
            color: #e4e6eb;
        }

        body.dark-mode .card,
        body.dark-mode .progress-section,
        body.dark-mode .results-section,
        body.dark-mode .summary-card {
            background-color: #242526;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }

        body.dark-mode .card-header,
        body.dark-mode .steel-table th {
            border-color: #3a3b3c;
        }

        body.dark-mode input[type="text"],
        body.dark-mode input[type="number"],
        body.dark-mode select {
            background-color: #3a3b3c;
            border-color: #3a3b3c;
            color: #e4e6eb;
        }

        body.dark-mode .steel-table th {
            background-color: #2d2e2f;
        }

        body.dark-mode .steel-table tr:hover td {
            background-color: #2d2e2f;
        }

        body.dark-mode .progress-container {
            background-color: #2d2e2f;
        }

        body.dark-mode .summary-card {
            background-color: #2d2e2f;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="header-content">
                <div>
                    <h1><i class="fas fa-industry"></i> 钢材优化系统</h1>
                    <div class="subtitle">最大化材料利用率，最小化损耗成本</div>
                </div>
                <button class="theme-switch" id="themeToggle">
                    <i class="fas fa-moon"></i> 深色模式
                </button>
            </div>
        </header>

        <div class="dashboard">
            <div class="card">
                <div class="card-header">
                    <h2 class="card-title"><i class="fas fa-ruler"></i> 设计钢材</h2>
                    <div class="card-actions">
                        <button class="btn btn-outline" id="importExcel">
                            <i class="fas fa-file-import"></i> 导入Excel
                        </button>
                        <button class="btn btn-primary" id="addDesignSteel">
                            <i class="fas fa-plus"></i> 添加钢材
                        </button>
                    </div>
                </div>
                
                <div class="input-group">
                    <input type="file" id="designFile" accept=".xlsx,.xls,.csv" style="display: none;">
                </div>
                
                <div class="table-container">
                    <table class="steel-table">
                        <thead>
                            <tr>
                                <th>编号</th>
                                <th>长度(mm)</th>
                                <th>数量</th>
                                <th>操作</th>
                            </tr>
                        </thead>
                        <tbody id="designTableBody">
                            <tr>
                                <td>A1-A5</td>
                                <td>2000</td>
                                <td>5</td>
                                <td>
                                    <button class="delete-btn">
                                        <i class="fas fa-trash"></i>
                                    </button>
                                </td>
                            </tr>
                            <tr>
                                <td>A6-A10</td>
                                <td>3000</td>
                                <td>5</td>
                                <td>
                                    <button class="delete-btn">
                                        <i class="fas fa-trash"></i>
                                    </button>
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <div class="card">
                <div class="card-header">
                    <h2 class="card-title"><i class="fas fa-cubes"></i> 模数钢材</h2>
                    <button class="btn btn-primary" id="addModuleSteel">
                        <i class="fas fa-plus"></i> 添加钢材
                    </button>
                </div>
                
                <div class="table-container">
                    <table class="steel-table">
                        <thead>
                            <tr>
                                <th>编号</th>
                                <th>长度(mm)</th>
                                <th>操作</th>
                            </tr>
                        </thead>
                        <tbody id="moduleTableBody">
                            <tr>
                                <td>B1</td>
                                <td>6000</td>
                                <td>
                                    <button class="delete-btn">
                                        <i class="fas fa-trash"></i>
                                    </button>
                                </td>
                            </tr>
                            <tr>
                                <td>B2</td>
                                <td>4000</td>
                                <td>
                                    <button class="delete-btn">
                                        <i class="fas fa-trash"></i>
                                    </button>
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <div class="card">
            <div class="card-header">
                <h2 class="card-title"><i class="fas fa-sliders-h"></i> 优化参数设置</h2>
            </div>
            
            <div class="settings-grid">
                <div class="input-group">
                    <label for="tolerance">允许误差 (mm)</label>
                    <input type="number" id="tolerance" value="5" min="0" step="0.1">
                </div>
                
                <div class="input-group">
                    <label for="cutLoss">切割损耗 (mm)</label>
                    <input type="number" id="cutLoss" value="3" min="0" step="0.1">
                </div>
                
                <div class="input-group">
                    <label for="weldLoss">焊接损耗 (mm)</label>
                    <input type="number" id="weldLoss" value="2" min="0" step="0.1">
                </div>
                
                <div class="input-group">
                    <label for="maxTime">最大计算时间 (秒)</label>
                    <input type="number" id="maxTime" value="300" min="10">
                </div>
                
                <div class="input-group">
                    <label for="targetLoss">期望损耗率 (%)</label>
                    <input type="number" id="targetLoss" value="10" min="0" max="100" step="0.1">
                </div>
                
                <div class="input-group">
                    <label for="density">材料密度 (kg/m³)</label>
                    <input type="number" id="density" value="7850">
                </div>
            </div>
        </div>

        <div class="progress-section">
            <div class="progress-header">
                <h2><i class="fas fa-tachometer-alt"></i> 优化进度</h2>
                <div class="status" id="statusText">准备就绪</div>
            </div>
            
            <div class="progress-container">
                <div class="progress-bar" id="progressBar" style="width: 0%"></div>
            </div>
            
            <div class="progress-info">
                <div>当前损耗率: <span id="currentLoss">0.0</span>%</div>
                <div>最佳损耗率: <span id="bestLoss">0.0</span>%</div>
                <div>计算时间: <span id="calcTime">0</span>秒</div>
            </div>
            
            <div class="control-group">
                <button class="btn btn-primary" id="startOptimization">
                    <i class="fas fa-play"></i> 开始优化
                </button>
                <button class="btn btn-warning" id="stopOptimization" disabled>
                    <i class="fas fa-stop"></i> 停止计算
                </button>
                <button class="btn btn-secondary" id="exportExcel" disabled>
                    <i class="fas fa-file-excel"></i> 导出Excel
                </button>
                <button class="btn btn-secondary" id="exportPDF" disabled>
                    <i class="fas fa-file-pdf"></i> 导出PDF
                </button>
            </div>
        </div>

        <div class="results-section">
            <div class="card-header">
                <h2 class="card-title"><i class="fas fa-chart-line"></i> 优化结果</h2>
                <div class="summary-text">最低损耗率: <span id="finalLossRate">8.2</span>%</div>
            </div>
            
            <div class="summary-cards">
                <div class="summary-card">
                    <h3>节省成本</h3>
                    <div class="value success">¥1,850</div>
                </div>
                <div class="summary-card">
                    <h3>材料利用率</h3>
                    <div class="value">91.8%</div>
                </div>
                <div class="summary-card">
                    <h3>模数钢材用量</h3>
                    <div class="value">42.5吨</div>
                </div>
                <div class="summary-card">
                    <h3>总组合数</h3>
                    <div class="value">18</div>
                </div>
            </div>
            
            <h3><i class="fas fa-table"></i> 组合详情</h3>
            <div class="table-container" style="overflow-x: auto;">
                <table class="steel-table">
                    <thead>
                        <tr>
                            <th>组合ID</th>
                            <th>设计钢材</th>
                            <th>设计总长(mm)</th>
                            <th>模数钢材</th>
                            <th>模数总长(mm)</th>
                            <th>差值(mm)</th>
                            <th>损耗率(%)</th>
                        </tr>
                    </thead>
                    <tbody id="resultsBody">
                        <tr>
                            <td>G1</td>
                            <td>A1, A2, A3</td>
                            <td>6500</td>
                            <td>B1 x 1</td>
                            <td>6000</td>
                            <td>500</td>
                            <td>8.3%</td>
                        </tr>
                        <tr>
                            <td>G2</td>
                            <td>A4, A5</td>
                            <td>3500</td>
                            <td>B2 x 1</td>
                            <td>4000</td>
                            <td>500</td>
                            <td>12.5%</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            
            <div class="chart-container">
                <canvas id="lossChart"></canvas>
            </div>
        </div>
        
        <footer>
            <p>钢材优化系统 © 2023 | 最大化材料利用率，降低生产成本</p>
        </footer>
    </div>
    
    <div class="notification" id="notification">
        <i class="fas fa-check-circle"></i>
        <div class="notification-content">
            <div class="notification-title">优化完成</div>
            <div class="notification-message">已找到最佳方案，损耗率8.2%</div>
        </div>
    </div>

    <script>
        // DOM Elements
        const themeToggle = document.getElementById('themeToggle');
        const designTableBody = document.getElementById('designTableBody');
        const moduleTableBody = document.getElementById('moduleTableBody');
        const progressBar = document.getElementById('progressBar');
        const statusText = document.getElementById('statusText');
        const startBtn = document.getElementById('startOptimization');
        const stopBtn = document.getElementById('stopOptimization');
        const exportExcelBtn = document.getElementById('exportExcel');
        const exportPDFBtn = document.getElementById('exportPDF');
        const notification = document.getElementById('notification');
        const importExcelBtn = document.getElementById('importExcel');
        const designFile = document.getElementById('designFile');
        
        // Toggle theme
        themeToggle.addEventListener('click', () => {
            document.body.classList.toggle('dark-mode');
            const isDark = document.body.classList.contains('dark-mode');
            themeToggle.innerHTML = isDark 
                ? '<i class="fas fa-sun"></i> 浅色模式' 
                : '<i class="fas fa-moon"></i> 深色模式';
        });
        
        // Show notification
        function showNotification(message, type = 'success') {
            const title = type === 'success' ? '优化完成' : '注意';
            notification.querySelector('.notification-title').textContent = title;
            notification.querySelector('.notification-message').textContent = message;
            notification.className = `notification ${type} show`;
            
            setTimeout(() => {
                notification.classList.remove('show');
            }, 5000);
        }
        
        // Simulate optimization progress
        function simulateOptimization() {
            let progress = 0;
            let currentLoss = 25;
            let bestLoss = 25;
            let time = 0;
            
            startBtn.disabled = true;
            stopBtn.disabled = false;
            statusText.textContent = "优化中...";
            
            const interval = setInterval(() => {
                progress += Math.random() * 5;
                currentLoss -= Math.random() * 0.8;
                time += 1;
                
                if (progress > 100) progress = 100;
                if (currentLoss < bestLoss) bestLoss = currentLoss;
                
                progressBar.style.width = `${progress}%`;
                document.getElementById('currentLoss').textContent = currentLoss.toFixed(1);
                document.getElementById('bestLoss').textContent = bestLoss.toFixed(1);
                document.getElementById('calcTime').textContent = time;
                
                if (progress >= 100) {
                    clearInterval(interval);
                    statusText.textContent = "优化完成";
                    startBtn.disabled = false;
                    stopBtn.disabled = true;
                    exportExcelBtn.disabled = false;
                    exportPDFBtn.disabled = false;
                    document.getElementById('finalLossRate').textContent = bestLoss.toFixed(1);
                    showNotification(`已找到最佳方案，损耗率${bestLoss.toFixed(1)}%`);
                    renderLossChart();
                }
            }, 200);
            
            stopBtn.addEventListener('click', () => {
                clearInterval(interval);
                statusText.textContent = "已停止";
                startBtn.disabled = false;
                stopBtn.disabled = true;
                exportExcelBtn.disabled = false;
                exportPDFBtn.disabled = false;
            });
        }
        
        // Initialize loss chart
        function renderLossChart() {
            const ctx = document.getElementById('lossChart').getContext('2d');
            
            // Simulated loss data
            const lossData = [];
            for (let i = 0; i < 20; i++) {
                lossData.push(25 - (i * 0.8) + (Math.random() - 0.5) * 2);
            }
            lossData[19] = 8.2; // Final value
            
            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: Array.from({length: 20}, (_, i) => i + 1),
                    datasets: [{
                        label: '损耗率变化',
                        data: lossData,
                        borderColor: '#3498db',
                        backgroundColor: 'rgba(52, 152, 219, 0.1)',
                        borderWidth: 3,
                        pointRadius: 4,
                        pointBackgroundColor: '#fff',
                        pointBorderWidth: 2,
                        tension: 0.3,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: false,
                            min: 5,
                            title: {
                                display: true,
                                text: '损耗率 (%)'
                            },
                            grid: {
                                color: 'rgba(0, 0, 0, 0.05)'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: '迭代次数'
                            },
                            grid: {
                                color: 'rgba(0, 0, 0, 0.05)'
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            display: false
                        },
                        tooltip: {
                            backgroundColor: 'rgba(0, 0, 0, 0.7)',
                            padding: 10,
                            cornerRadius: 4,
                            displayColors: false
                        }
                    }
                }
            });
        }
        
        // Event listeners
        startBtn.addEventListener('click', simulateOptimization);
        
        importExcelBtn.addEventListener('click', () => {
            designFile.click();
        });
        
        designFile.addEventListener('change', function() {
            if (this.files.length > 0) {
                showNotification('设计钢材数据已导入', 'success');
            }
        });
        
        // Initialize chart
        window.onload = renderLossChart;
    </script>
</body>
</html>
