from src.models.repositories.user_repository import UserRepository

from src.services.users.create_user_service import CreateUserService

from src.controllers.users.create_user_controller import CreateUserController

def make_create_user_controller() -> CreateUserController:
    """
    Factory para criar uma instância de CreateUserController
    com suas dependências injetadas.
    
    Returns:
        CreateUserController: Instância do controlador de criação de usuários
    """
    user_repository = UserRepository()
    create_user_service = CreateUserService(user_repository)
    create_user_controller = CreateUserController(create_user_service)
    
    return create_user_controller
