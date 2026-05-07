# %% [markdown]
# #### Fastapi-Learing: uv add fastapi uvicorn-> #测试运行api的服务器

# %%
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
  text: str = None
  is_done: bool = False

items = []

@app.get("/notebook")
def root():
    return {"Hello": "World"}

# 👇 改成支持 query 参数的 POST，就不会卡死了
@app.post("/notebook/items", status_code=201)  
def create_item(item: Item):
    items.append(item)
    return items

#服务器得到多个    
@app.get("/notebook/items",response_model=List[Item])
def list_items(limit: int = 10):
  return items[0:limit]

#得到一个，否则HTTPException返回错误
@app.get("/items/{item_id}", response_model=Item)
def get_items(item_id: int) -> Item:
    if item_id < len(items):
      return items[item_id]
    else:
      raise HTTPException(status_code=404, detail = f"Item {item_id} not found")  

# %% [markdown]
# #运行服务器
# uvicorn notebook.fastapi_learn:app --reload  #--reload文件更新时自动刷新服务器
# http://127.0.0.1:8000/notebook 打开 

# curl.exe -X POST -H "Content-Type: application/json" 'http://127.0.0.1:8000/notebook/items?item=apple' post添加

# curl.exe -X GET http://127.0.0.1:8000/items/0 get添加

# pydantic 的 BaseModel 规定post进入服务器的格式 创建验证

# curl.exe -X POST -H "Content-Type: application/json" -d '{\"text\":\"apple\"}' "http://127.0.0.1:8000/notebook/items"      规定格式post



# 1. curl.exe
# 作用：在 Windows PowerShell 里发送网络请求的工具
# 意思：我要发送一个请求给服务器
# 2. -X POST
# 作用：指定请求方式为 POST
# 意思：我要提交 / 添加数据（不是获取数据）
# 3. -H "Content-Type: application/json"
# 作用：请求头，告诉服务器我发的是 JSON 格式
# 意思：我给你的数据是 JSON 格式，请按 JSON 解析
# 4. -d '{"text":"apple"}'
# 作用：要发送的数据内容（请求体）
# 意思：
# json
# {
#   "text": "apple"
# }
# 我给你传一个字段叫 text，值是 apple
# 5. 'http://127.0.0.1:8000/notebook/items'
# 作用：请求地址
# 意思：把数据发送到这个接口
