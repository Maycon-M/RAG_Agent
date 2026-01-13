from src.models.repositories.student_repository import StudentRepository

from src.services.students.create_studend_service import CreateStudentService

from src.controllers.students.create_student_controller import CreateStudentController

def make_create_student_controller() -> CreateStudentController:
    """
    Factory para criar uma instância de CreateStudentController
    com suas dependências injetadas.
    
    Returns:
        CreateStudentController: Instância do controlador de criação de alunos
    """
    student_repository = StudentRepository()
    create_student_service = CreateStudentService(student_repository)
    create_student_controller = CreateStudentController(create_student_service)
    
    return create_student_controller
