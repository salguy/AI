from pydantic import BaseModel

class AIInput(BaseModel):
    input_text: str
    scheduleId: int
    