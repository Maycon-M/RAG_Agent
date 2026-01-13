from __future__ import annotations

from abc import ABC, abstractmethod

from sqlalchemy.orm import Session

from src.domain.requests.students.student_create_request import StudentCreateRequest
from src.domain.responses.students.student_create_response import StudentCreateResponse

class CreateStudentServiceInterface(ABC):
    
    """ 
    Serviço para criação de alunos.
    """
    
    @abstractmethod
    async def create_student(self, db: Session, request: StudentCreateRequest) -> StudentCreateResponse:
        """
        Cria um novo aluno.
        
        Args:
            db: Sessão do banco de dados
            request: Dados do aluno a ser criado
            
        Returns:
            Student: Aluno criado
        """
        raise NotImplementedError()
