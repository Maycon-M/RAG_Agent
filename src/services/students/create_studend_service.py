from __future__ import annotations

from uuid import uuid4

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from src.interfaces.repositories.student_repository_interface import StudentRepositoryInterface
from src.interfaces.services.students.create_studend_service_interface import CreateStudentServiceInterface

from src.domain.requests.students.student_create_request import StudentCreateRequest
from src.domain.responses.students.student_create_response import StudentCreateResponse

from src.models.entities.student import Student

from src.errors.domain.already_existing import AlreadyExistingError
from src.errors.domain.sql_error import SqlError

from src.core.logging_config import get_logger

class CreateStudentService(CreateStudentServiceInterface):
    
    """ 
    Serviço para criação de alunos.
    """
    
    def __init__(self, repository: StudentRepositoryInterface) -> None:
        self.__repository = repository
        self.__logger = get_logger(__name__)
        
    async def create_student(self, db: Session, request: StudentCreateRequest) -> StudentCreateResponse:
        """
        Cria um novo aluno.
        
        Args:
            db: Sessão do banco de dados
            request: Dados do aluno a ser criado
            
        Returns:
            Student: Aluno criado
        """
        try:
            self.__logger.info("Criando novo aluno: %s", request.full_name)
            
            student = self.__repository.create(
                db,
                uuid=uuid4(),
                full_name=request.full_name,
                email=request.email,
                registration_number=request.registration_number
            )
            
            self.__logger.info("Aluno criado com sucesso: %s", student.uuid)
            return self._format_response(student)
        
        except IntegrityError as e:
            self.__logger.error("Erro de integridade ao criar aluno: %s", e, exc_info=True)
            raise AlreadyExistingError(
                message="Matricula já está cadastrada",
                context={"matricula": request.registration_number},
                cause=e
            ) from e
        
        except Exception as e:
            self.__logger.error("Erro inesperado ao criar aluno: %s", e)
            raise SqlError(
                message="Erro ao criar aluno no banco de dados",
                context={"nome": request.full_name},
                cause=e
            ) from e

    def _format_response(self, student: Student) -> StudentCreateResponse:
        """
        Formata a resposta do aluno criado.
        
        Args:
            student: Entidade Student
            
        Returns:
            StudentCreateResponse: Resposta formatada
        """
        return StudentCreateResponse.model_validate(student)
