from typing import List
from pydantic import BaseModel


class NamedUrl(BaseModel):
    name: str
    url: str


class PositionInfo(BaseModel):
    answer:str
    references:List[NamedUrl]


