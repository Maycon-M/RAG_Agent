from fastapi import APIRouter, Request, Body, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from src.domain.http.http_request import HttpRequest
from src.domain.http.http_response import HttpResponse
from src.domain.http.caller_domains import CallerMeta

# Request Models
from src.domain.requests.students.student_create_request import StudentCreateRequest

# Response Models
from src.domain.responses.students.student_create_response import StudentCreateResponse

from src.core.logging_config import get_logger

from src.main.composer.students_composer import (
    make_create_student_controller,
)

from src.main.dependencies.request_meta import get_caller_meta
from src.main.dependencies.get_db_session import get_db
from src.main.dependencies.auth_jwt import auth_jwt_verify

logger = get_logger(__name__)

router = APIRouter(
    prefix="/students",
    tags=["Students"],
)

@router.post(
    "/register",
    response_model=StudentCreateResponse,
    status_code=201,
)
def register_student(
    request: Request,
    body: StudentCreateRequest = Body(...),
    caller: CallerMeta = Depends(get_caller_meta),
    db:Session=Depends(get_db),
):
    """
    Endpoint para registrar um novo aluno.
    
    Args:
        request (Request): Objeto de requisição FastAPI
        body (StudentCreateRequest): Corpo da requisição contendo os dados do aluno
        caller (CallerMeta): Metadados do chamador (injetado via dependência)
        db (Session): Sessão do banco de dados (injetado via dependência)
        
    Returns:
        JSONResponse: Resposta HTTP com os dados do aluno criado
    """
    try:
        try:
            auth_header = request.headers.get("Authorization") or ""
            token = auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else ""
            token_infos = auth_jwt_verify(token, db, scope="teacher")
            logger.debug("Token Valido")
        except HTTPException as e:
            logger.error("Erro de autenticação: %s",  str(e.detail))
            raise HTTPException(status_code=e.status_code, detail=e.detail) from e
        except Exception as e:
            logger.error("Erro inesperado de autenticação: %s", str(e))
            raise HTTPException(status_code=500, detail="Erro interno de autenticação") from e
        
        http_request = HttpRequest(
            body=body,
            db=db,
            caller=caller,
            token_infos=token_infos,
            headers=request.headers,
        )
        
        controller = make_create_student_controller()
        http_response: HttpResponse = controller.handle(http_request)
        
        return JSONResponse(
            status_code=http_response.status_code,
            content=http_response.body.model_dump(mode='json') if http_response.body else None
        )
        
    except HTTPException as e:
        logger.error("Erro HTTP: %s", str(e.detail))
        raise HTTPException(status_code=e.status_code, detail=e.detail) from e
    except Exception as e:
        logger.error("Erro inesperado: %s", str(e))
        raise HTTPException(status_code=500, detail="Erro interno do servidor") from e
