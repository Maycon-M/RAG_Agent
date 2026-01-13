from __future__ import annotations

import asyncio
from fastapi import HTTPException

from src.domain.http.http_request import HttpRequest
from src.domain.http.http_response import HttpResponse

from src.interfaces.controllers.controllers_interface import ControllerInterface
from src.interfaces.services.students.create_studend_service_interface import CreateStudentServiceInterface

from src.errors.domain.sql_error import SqlError

from src.core.logging_config import get_logger

class CreateStudentController(ControllerInterface):
    """  
    Controller que delega ao CreateStudentService a criação de um novo aluno.
    """
    
    def __init__(self, service: CreateStudentServiceInterface) -> None:
        self.__service = service
        self.__logger = get_logger(__name__)
        
    def handle(self, http_request: HttpRequest) -> HttpResponse:
        """
        Processa a requisição de criação de aluno.
        
        Args:
            http_request: Requisição HTTP contendo os dados do aluno
            
        Returns:
            HttpResponse: Resposta HTTP com os dados do aluno criado
            
        Raises:
            HTTPException: Em caso de erro (409 para matrícula duplicada, 500 para erros de BD)
        """
        
        db = http_request.db
        caller = http_request.caller
        
        self.__logger.debug(
            "Handling create student request from caller: %s - %s - %s", 
            caller.caller_app if caller else "unknown",
            caller.caller_user if caller else "unknown",
            caller.ip if caller else "unknown"
        )
        
        try:
            # O body já vem validado como StudentCreateRequest pela rota FastAPI
            request = http_request.body
            
            # Delega ao serviço (agora async)
            result = asyncio.run(self.__service.create_student(db, request))
            
            self.__logger.info("Aluno criado com sucesso: %s", result.uuid)
            
            return HttpResponse(
                status_code=201,
                body=result
            )
        
        except SqlError as sql_err:
            self.__logger.error("Erro de banco de dados ao criar aluno: %s", sql_err, exc_info=True)
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Erro de banco de dados ao criar aluno",
                    "code": sql_err.code
                }
            ) from sql_err
        
        except ValueError as val_err:
            self.__logger.error("Erro de valor ao criar aluno: %s", val_err, exc_info=True)
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "Dados inválidos",
                    "message": str(val_err)
                }
            ) from val_err
        
        except Exception as exc:
            self.__logger.error("Erro inesperado ao criar aluno: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=500,
                detail={"error": "Erro interno no servidor"}
            ) from exc
