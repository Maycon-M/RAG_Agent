from __future__ import annotations

from typing import Dict, Any, Optional
from uuid import UUID

import httpx

from src.core.settings import settings
from src.core.logging_config import get_logger


class BrevoHandler:
    """
    Handler para integração com a API da Brevo (SendInBlue).
    
    Responsável por:
    - Criar/atualizar contatos na Brevo
    - Enviar emails transacionais via templates
    """
    
    def __init__(self):
        self.__logger = get_logger(__name__)
        self.__api_key = settings.BREVO_API_KEY
        self.__base_url = "https://api.brevo.com/v3"
        self.__headers = {
            "accept": "application/json",
            "api-key": self.__api_key,
            "content-type": "application/json"
        }
    
    async def create_or_update_contact(
        self, 
        email: str, 
        user_uuid: UUID,
        user_type: str,
        attributes: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Cria ou atualiza um contato na Brevo.
        
        Args:
            email: Email do contato
            user_uuid: UUID do usuário no banco de dados
            user_type: Tipo do usuário (admin, teacher, student)
            attributes: Atributos adicionais do contato
            
        Returns:
            bool: True se sucesso, False se falha
        """
        url = f"{self.__base_url}/contacts"
        
        payload = {
            "email": email,
            "attributes": {
                "UUID_DB": str(user_uuid),
                "USER_TYPE": user_type,
                **(attributes or {})
            },
            "updateEnabled": True
        }
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(url, json=payload, headers=self.__headers)
                
                if response.status_code in [201, 204]:
                    self.__logger.info(
                        "Contato criado/atualizado na Brevo: email=%s, uuid=%s",
                        email, user_uuid
                    )
                    return True
                self.__logger.error(
                    "Erro ao criar contato na Brevo: status=%s, response=%s",
                    response.status_code, response.text
                )
                return False
                    
        except httpx.RequestError as e:
            self.__logger.error("Erro de conexão com Brevo API: %s", str(e), exc_info=True)
            return False
        except Exception as e:
            self.__logger.error("Erro inesperado ao criar contato na Brevo: %s", str(e), exc_info=True)
            return False
    
    async def send_verification_email(
        self, 
        email: str, 
        user_uuid: UUID,
        user_name: Optional[str] = None
    ) -> bool:
        """
        Envia email de verificação usando template da Brevo.
        
        Args:
            email: Email do destinatário
            user_uuid: UUID do usuário (usado no template)
            user_name: Nome do usuário (opcional)
            
        Returns:
            bool: True se sucesso, False se falha
        """
        url = f"{self.__base_url}/smtp/email"
        
        payload = {
            "to": [{"email": email, "name": user_name or email}],
            "templateId": settings.BREVO_VERIFICATION_TEMPLATE_ID,
            "params": {
                "USER_NAME": user_name or email,
                "UUID_DB": str(user_uuid)
            }
        }
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(url, json=payload, headers=self.__headers)
                
                if response.status_code == 201:
                    self.__logger.info(
                        "Email de verificação enviado via Brevo: email=%s, uuid=%s",
                        email, user_uuid
                    )
                    return True
                self.__logger.error(
                    "Erro ao enviar email via Brevo: status=%s, response=%s",
                    response.status_code, response.text
                )
                return False
                    
        except httpx.RequestError as e:
            self.__logger.error("Erro de conexão com Brevo API: %s", str(e), exc_info=True)
            return False
        except Exception as e:
            self.__logger.error("Erro inesperado ao enviar email via Brevo: %s", str(e), exc_info=True)
            return False
