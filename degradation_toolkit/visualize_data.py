import json
import os
import random
import sys
from pathlib import Path

import pandas
from data_reader import read_img_general
from flask import Flask, make_response, request

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index_page():
    ps = '''
<a href="/leo/0">leonardo</a> <br />
'''
    return f"""<body>
    Subsets
  <p>
  {ps}
</p>
</body>"""


@app.route("/<name>/<int:page>/")
def show_page(name, page):
    table_rows = ""

    data = []

    if name == 'leo':
        with open('omnigen/data_json/meta_info_imageEdit_ultraEdit_4M_part2.json') as f:
            data = json.load(f)

    st, ed = page * 20, (page + 1) * 20
    for d in data[st:ed]:
        input_image = f"""<img src="/imgs/{d['input_path']}" style="width: 400px">"""
        target_image = f"""<img src="/imgs/{d['target_path']}" style="width: 400px">"""

        caption = ''
        for k, v in d.items():
            caption += f'{k}: {v}\n\n'

        table_rows += f"""<tr>
    <td style="text-align: center;vertical-align: middle;">{input_image}</td>
    <td style="text-align: center;vertical-align: middle;">{target_image}</td>
    <td><pre style="white-space: pre-wrap;width: 400px;">{caption}</pre></td>
</tr>"""

    return f"""<head>
    <style>
    table, th, td {{
        border: 1px solid black;
        border-collapse: collapse;
        padding-left: 10px;
        padding-right: 10px;
    }};
    </style>
</head>
<body>
    <p><a href="/">主页</a> |
    <a href="/{name}/{max(page - 1, 0)}">上一页</a> | 
    <a href="/{name}/{page + 1}">下一页</a>
    <table>
    <tr>
        <th style="width: 400px">输入图像</th>
        <th style="width: 400px">目标图像</th>
        <th>注释</th>
    </tr>
    {table_rows}
    </table>
    <a href="/">主页</a> |
    <a href="/{name}/{max(page - 1, 0)}">上一页</a> | 
    <a href="/{name}/{page + 1}">下一页</a>
</body>
"""


@app.route("/imgs/<path:path>")
def show_img(path):
    try:
        # 调用 read_img_general 函数
        img_data = read_img_general(path)

        # 检查是否读取成功
        if img_data:
            response = make_response(img_data)
            response.headers.set('Content-Type', 'image/jpeg')  # 假设图片是 JPEG 格式
            return response
        else:
            return "未找到图像", 404
    except Exception as e:
        # 打印错误信息并返回错误响应
        print(f"处理图像 {path} 时出错: {e}")
        return f"处理图像时出错: {str(e)}", 500


app.run(
    host="127.0.0.1",
    port=7848,
    debug=True,
)
