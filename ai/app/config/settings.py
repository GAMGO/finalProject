import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import Field
load_dotenv()

dataVITE_BASE_URL: str = Field(..., env="DATAVITE_BASE_URL")
openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    
class Settings(BaseSettings):
    
    dataVITE_BASE_URL: str  # ⭐ 반드시 필요

 
    
    DB_HOST: str = "localhost"
    DB_USER: str = "root"
    DB_PASSWORD: str = "1234"
    DB_NAME: str = "my_project_db"

    OPENAI_API_KEY: str = ""

    class Config:
        env_file = ".env"
        extra = "allow" 