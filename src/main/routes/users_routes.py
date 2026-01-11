from fastapi import APIRouter, Depends, HTTPException, Request, Body
from fastapi.responses import JSONResponse

from src.domain.http.http_request import HttpRequest
from src.domain.http.http_response import HttpResponse
from src.domain.http.caller_domains import CallerMeta

# Request Models
from src.domain.requests.users.user_create_request import UserCreateRequest

# Response Models
from src.domain.responses.users.user_response import UserResponse

from src.core.logging_config import get_logger

from src.main.composer.users_composer import (
    make_create_user_controller,
)

from src.main.dependencies.request_meta import get_caller_meta
from src.main.dependencies.get_db_session import get_db

logger = get_logger(__name__)

router = APIRouter(
    prefix="/users",
    tags=["Users"],
)

@router.post(
    "/register",
    response_model=UserResponse,
    status_code=201
)
def register_user(
    request: Request,
    body: UserCreateRequest = Body(...),
    caller: CallerMeta = Depends(get_caller_meta),
    db=Depends(get_db),
):
    """
    Endpoint para registrar um novo usuário.
    
    Args:
        request (Request): Objeto de requisição FastAPI
        body (UserCreateRequest): Corpo da requisição contendo os dados do usuário
        caller (CallerMeta): Metadados do chamador (injetado via dependência)
        db (Session): Sessão do banco de dados (injetado via dependência)
        
    Returns:
        JSONResponse: Resposta HTTP com os dados do usuário criado
    """
    http_request = HttpRequest(
        body=body,
        db=db,
        caller=caller,
        headers=request.headers,
    )
    
    controller = make_create_user_controller()
    
    try:
        http_response: HttpResponse = controller.handle(http_request)
        return JSONResponse(
            status_code=http_response.status_code,
            content=http_response.body
        )
    except HTTPException as e:
        logger.error("Erro ao registrar usuário: %s", str(e.detail))
        raise e
    except Exception as e:
        logger.exception("Erro inesperado ao registrar usuário")
        raise HTTPException(status_code=500, detail="Erro interno do servidor") from e
