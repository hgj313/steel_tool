from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from optimizer import SteelOptimizer
import sqlite3
import os
import time
import pandas as pd
from fpdf import FPDF
import io

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 数据库初始化
def init_db():
    conn = sqlite3.connect('data/database.db')
    c = conn.cursor()
    
    # 创建设计钢材表
    c.execute('''CREATE TABLE IF NOT EXISTS design_steels (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 length REAL NOT NULL,
                 quantity INTEGER NOT NULL
                 )''')
    
    # 创建模数钢材表
    c.execute('''CREATE TABLE IF NOT EXISTS module_steels (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 length REAL NOT NULL
                 )''')
    
    # 创建优化结果表
    c.execute('''CREATE TABLE IF NOT EXISTS optimization_results (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                 loss_rate REAL,
                 cost_saving REAL,
                 calc_time REAL
                 )''')
    
    # 创建组合详情表
    c.execute('''CREATE TABLE IF NOT EXISTS combination_details (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 result_id INTEGER,
                 group_id TEXT,
                 design_steels TEXT,
                 design_length REAL,
                 module_steels TEXT,
                 module_length REAL,
                 difference REAL,
                 loss_rate REAL,
                 FOREIGN KEY(result_id) REFERENCES optimization_results(id)
                 )''')
    
    conn.commit()
    conn.close()

# 确保数据目录存在
os.makedirs('data/uploads', exist_ok=True)
os.makedirs('data/results', exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_file():
    """处理设计钢材文件上传"""
    if 'file' not in request.files:
        return jsonify({'error': '没有文件上传'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '未选择文件'}), 400
    
    try:
        # 保存文件
        filename = os.path.join('data/uploads', file.filename)
        file.save(filename)
        
        # 处理Excel/CSV文件
        if filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(filename)
        else:  # CSV
            df = pd.read_csv(filename)
        
        # 验证数据格式
        if '长度(mm)' not in df.columns or '数量' not in df.columns:
            return jsonify({'error': '文件格式错误，需要包含"长度(mm)"和"数量"列'}), 400
        
        # 保存到数据库
        conn = sqlite3.connect('data/database.db')
        c = conn.cursor()
        c.execute('DELETE FROM design_steels')  # 清除旧数据
        
        for _, row in df.iterrows():
            c.execute('INSERT INTO design_steels (length, quantity) VALUES (?, ?)',
                      (row['长度(mm)'], row['数量']))
        
        conn.commit()
        conn.close()
        
        return jsonify({'message': '文件上传成功', 'count': len(df)})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/optimize', methods=['POST'])
def optimize():
    """启动优化计算"""
    try:
        # 获取参数
        data = request.json
        params = {
            'tolerance': float(data.get('tolerance', 5)),
            'cut_loss': float(data.get('cut_loss', 3)),
            'weld_loss': float(data.get('weld_loss', 2)),
            'max_time': int(data.get('max_time', 300)),
            'target_loss': float(data.get('target_loss', 10))
        }
        
        # 从数据库获取设计钢材
        conn = sqlite3.connect('data/database.db')
        c = conn.cursor()
        c.execute('SELECT id, length, quantity FROM design_steels')
        design_steels = [{'id': row[0], 'length': row[1], 'quantity': row[2]} for row in c.fetchall()]
        
        # 获取模数钢材
        c.execute('SELECT id, length FROM module_steels')
        module_steels = [{'id': row[0], 'length': row[1]} for row in c.fetchall()]
        
        conn.close()
        
        if not design_steels:
            return jsonify({'error': '没有设计钢材数据'}), 400
        
        # 创建优化器并运行
        optimizer = SteelOptimizer(design_steels, module_steels, params)
        result = optimizer.optimize()
        
        # 保存结果到数据库
        conn = sqlite3.connect('data/database.db')
        c = conn.cursor()
        
        # 保存优化结果摘要
        c.execute('''INSERT INTO optimization_results 
                     (loss_rate, cost_saving, calc_time) 
                     VALUES (?, ?, ?)''',
                  (result['loss_rate'], result['cost_saving'], result['calc_time']))
        result_id = c.lastrowid
        
        # 保存组合详情
        for group in result['combinations']:
            c.execute('''INSERT INTO combination_details 
                         (result_id, group_id, design_steels, design_length, 
                          module_steels, module_length, difference, loss_rate)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                      (result_id, group['group_id'], 
                       ','.join(group['design_steels']), group['design_length'],
                       ','.join([f"{m['id']}:{m['count']}" for m in group['module_steels']]), 
                       group['module_length'], group['difference'], group['loss_rate']))
        
        conn.commit()
        conn.close()
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/export/excel/<result_id>')
def export_excel(result_id):
    """导出Excel结果"""
    try:
        # 从数据库获取结果
        conn = sqlite3.connect('data/database.db')
        c = conn.cursor()
        
        # 获取结果摘要
        c.execute('SELECT * FROM optimization_results WHERE id = ?', (result_id,))
        result = c.fetchone()
        if not result:
            return jsonify({'error': '结果不存在'}), 404
        
        # 获取组合详情
        c.execute('SELECT * FROM combination_details WHERE result_id = ?', (result_id,))
        details = c.fetchall()
        
        # 创建DataFrame
        columns = ['组合ID', '设计钢材', '设计总长(mm)', '模数钢材', '模数总长(mm)', '差值(mm)', '损耗率(%)']
        data = []
        
        for detail in details:
            data.append([
                detail[2],  # group_id
                detail[3],  # design_steels
                detail[4],  # design_length
                detail[5],  # module_steels
                detail[6],  # module_length
                detail[7],  # difference
                f"{detail[8]:.2f}%"  # loss_rate
            ])
        
        df = pd.DataFrame(data, columns=columns)
        
        # 创建Excel文件
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='优化结果', index=False)
            
            # 添加摘要信息
            workbook = writer.book
            worksheet = workbook.add_worksheet('摘要')
            
            summary_data = [
                ['最低损耗率', f"{result[2]:.2f}%"],
                ['节省成本', f"¥{result[3]:.2f}"],
                ['计算时间', f"{result[4]:.1f}秒"]
            ]
            
            for row, (label, value) in enumerate(summary_data):
                worksheet.write(row, 0, label)
                worksheet.write(row, 1, value)
        
        output.seek(0)
        
        # 返回Excel文件
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=f'钢材优化结果_{result_id}.xlsx'
        )
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/export/pdf/<result_id>')
def export_pdf(result_id):
    """导出PDF结果"""
    try:
        # 从数据库获取结果
        conn = sqlite3.connect('data/database.db')
        c = conn.cursor()
        
        # 获取结果摘要
        c.execute('SELECT * FROM optimization_results WHERE id = ?', (result_id,))
        result = c.fetchone()
        if not result:
            return jsonify({'error': '结果不存在'}), 404
        
        # 获取组合详情
        c.execute('SELECT * FROM combination_details WHERE result_id = ?', (result_id,))
        details = c.fetchall()
        
        # 创建PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "钢材优化结果报告", 0, 1, 'C')
        pdf.ln(10)
        
        # 添加摘要
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "优化结果摘要", 0, 1)
        pdf.set_font("Arial", '', 12)
        
        summary = [
            f"最低损耗率: {result[2]:.2f}%",
            f"节省成本: ¥{result[3]:.2f}",
            f"计算时间: {result[4]:.1f}秒"
        ]
        
        for item in summary:
            pdf.cell(0, 10, item, 0, 1)
        
        pdf.ln(10)
        
        # 添加组合详情
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "组合详情", 0, 1)
        pdf.set_font("Arial", '', 10)
        
        # 表头
        headers = ['组合ID', '设计钢材', '设计总长', '模数钢材', '模数总长', '差值', '损耗率']
        col_widths = [20, 40, 25, 40, 25, 20, 20]
        
        for i, header in enumerate(headers):
            pdf.cell(col_widths[i], 10, header, 1, 0, 'C')
        pdf.ln()
        
        # 表格内容
        for detail in details:
            pdf.cell(col_widths[0], 10, detail[2], 1)  # group_id
            pdf.cell(col_widths[1], 10, detail[3][:30], 1)  # design_steels
            pdf.cell(col_widths[2], 10, str(detail[4]), 1, 0, 'R')  # design_length
            pdf.cell(col_widths[3], 10, detail[5][:30], 1)  # module_steels
            pdf.cell(col_widths[4], 10, str(detail[6]), 1, 0, 'R')  # module_length
            pdf.cell(col_widths[5], 10, str(detail[7]), 1, 0, 'R')  # difference
            pdf.cell(col_widths[6], 10, f"{detail[8]:.2f}%", 1, 0, 'R')  # loss_rate
            pdf.ln()
        
        # 保存到内存
        pdf_output = pdf.output(dest='S').encode('latin1')
        
        return send_file(
            io.BytesIO(pdf_output),
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'钢材优化报告_{result_id}.pdf'
        )
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    init_db()
    app.run(host='127.0.0.1', port=5000, debug=True)
