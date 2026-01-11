from src.models.repositories.user_repository import UserRepository
from src.models.repositories.auth_refresh_token_repository import AuthRefreshTokenRepository

from src.services.users.create_user_service import CreateUserService
from src.services.users.verify_email_service import VerifyEmailService

from src.controllers.users.create_user_controller import CreateUserController
from src.controllers.users.verify_email_controller import VerifyEmailController

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

def make_verify_email_controller() -> VerifyEmailController:
    """
    Factory para criar uma instância de VerifyEmailController
    com suas dependências injetadas.
    
    Returns:
        VerifyEmailController: Instância do controlador de verificação de email
    """
    user_repository = UserRepository()
    refresh_token_repository = AuthRefreshTokenRepository()
    verify_email_service = VerifyEmailService(user_repository, refresh_token_repository)
    verify_email_controller = VerifyEmailController(verify_email_service)
    
    return verify_email_controller
