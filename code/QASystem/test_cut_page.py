# flask_pagination最后实际上就是之前设想的动态构建html的方式（拼接各个模块的html字符串就能实现），效果就是在html页面中增加一个实现分页的html模块，默认语法是boostrap，因此直接按调整boostrap风格的方式调整其分页风格即可
# 现在主要的问题是禁用状态下的分页组件（包括前后翻页和中间的省略号）的样式怎么调整（好像是有边框的，希望能把边框去掉
# 因为禁用状态(disabled)下标签正好都是<span>类的，目前可以通过定义<span>类的样式实现对禁用状态的控制，但禁用状态本身还是无法调整（同一类元素，如果一个class里有disabled，另一个没有，两个元素的样式是一样的，按目前的方式为disabled的元素设置样式并不生效
# 应该还是指定元素的方法有问题
from flask import Flask, render_template, request
from flask_paginate import Pagination
#from RAGoT.main import perform_qa
app = Flask(__name__)

# 测试分页
@app.route('/test')
def page_test():
    # 获取数据库中的数据（使用connMysql.py中写好的方法来查）
    # 本项目通过保存json文件（相当于和原始数据集一样的结构）管理数据  只有第一次调用时会调用问答系统进行查询，只要能找到对应的文件（怎么找？？？文件具体怎么管理？感觉可以用问题和哈希值对应的方式）就直接返回
    # 感觉还是不太好
    # 但为了实现分页，目前只能考虑这种方式。或者就是生成的页面就一次性包含了所有数据，不用每次进入某一页时读取相应的数据，但现有的分页方法好像都不支持这种情况
    content = [("Big Picture (magazine)","a"), ("Picture Play (magazine)","b"), ("Warren Publishing","c"), ("The Big Picture","d"), ("Current Publishing (UK)","e"), ("Highline Big Picture","f"), ("James Brown (publisher)","j"), ("James Warren (politician)","h"), ("Diane Warren","i"), ("1984 (magazine)","g"), ("Jamin Warren","k"),("Big Picture (magazine)","l"), ("Picture Play (magazine)","m"), ("Warren Publishing","n"), ("The Big Picture","o"), ("Current Publishing (UK)","p"), ("Highline Big Picture","q"), ("James Brown (publisher)","r"), ("James Warren (politician)","s"), ("Diane Warren","t"), ("1984 (magazine)","u"), ("Jamin Warren","v")]
    # print(len(content), type(content), content)		# 测试输出
    # 每一页显示记录数
    pageSize = 2
	
    page = request.args.get('page', 1, type=int)
    # 为了处理用户输入的超出页码范围的数字，添加以下代码
    if page > len(content) or page < 1:
        page = 1

    # 对获取到的数据进行切片
    start = (page - 1) * pageSize	# 开始，每一页开始位置
    end = start + pageSize			# 结束，每一页结束位置
    slices = slice(start, end)
    slicontent = content[slices]	# 切片

    """
    query: 我们要分页的集合对象，content为要分页的对象
    page:当前请求的页码
    per_page:每页的数据个数，自定义
    total:数据总量，也就是共有19条数据
    items:当前页需要显示的数据，因为页码是从1开始的，而列表的索引是从0开始的，所以要处理好这种转化关系。我们的例子是每页只显示5条数据，还比较好计算，如果是多条数据，计算的时候要细心一些
    """
    # 下面就是得到的某一页的分页对象
    current_page = Pagination(content, page=page, per_page=pageSize, total=len(content), items=content[page-1],prev_label="<i class='iconfont' style='font-size: 20px'>&#xeb62;</i>",next_label="<i class='iconfont' style='font-size: 20px'>&#xeb62;</i>")

    total_page = current_page.total     # 共有几条数据

    context = {
        'current_page': current_page,           # 获取到的数据库的数据
        'total_page': total_page,     # 共有几条数据
        'slicontent': slicontent,		# 数据切片显示
    }

    return render_template("cut_pages.html", **context)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)