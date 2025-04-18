"""
�����Ȣ#nAPI���ݤ��
"""

from typing import Any, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core.dependencies import get_current_user
from app.db.database import get_db
from app.models.schemas.project import (
    Project, 
    ProjectCreate, 
    ProjectUpdate, 
    ProjectList
)
# ,��go��n��ӹ���
# from app.services.project_service import project_service


router = APIRouter()


@router.get("", response_model=ProjectList)
def get_projects(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    name: Optional[str] = None,
) -> Any:
    """
    ������ ��֗Y�
    
    - �L@	Y�������hl������Ȓ֗
    - name�����g������gգ�����
    """
    # �ï�����Y��n��goDBK�֗	
    mock_items = [
        {
            "id": "123e4567-e89b-12d3-a456-426614174000",
            "name": "q�~�����",
            "description": "q�~gn���������Y�������",
            "is_public": True,
            "user_id": current_user["id"],
            "created_at": "2025-04-01T09:00:00",
            "updated_at": "2025-04-01T09:00:00"
        },
        {
            "id": "223e4567-e89b-12d3-a456-426614174001",
            "name": "_n�����",
            "description": "_n�gn��������Y�������",
            "is_public": False,
            "user_id": current_user["id"],
            "created_at": "2025-04-05T10:30:00",
            "updated_at": "2025-04-06T14:20:00"
        }
    ]
    
    # name�����gգ���
    if name:
        mock_items = [item for item in mock_items if name.lower() in item["name"].lower()]
    
    return {
        "items": mock_items,
        "total": len(mock_items),
        "skip": skip,
        "limit": limit
    }


@router.post("", response_model=Project, status_code=status.HTTP_201_CREATED)
def create_project(
    *,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
    project_in: ProjectCreate,
) -> Any:
    """�������Ȓ\Y�"""
    # ,��goproject_service�cf�����Ȓ\
    # project = project_service.create(db, obj_in=project_in, user_id=current_user["id"])
    
    # �ï�����Y
    return {
        "id": "323e4567-e89b-12d3-a456-426614174002",
        "name": project_in.name,
        "description": project_in.description,
        "is_public": project_in.is_public,
        "user_id": current_user["id"],
        "created_at": "2025-04-18T12:00:00",
        "updated_at": "2025-04-18T12:00:00"
    }


@router.get("/{project_id}", response_model=Project)
def get_project(
    *,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
    project_id: UUID,
) -> Any:
    """�W_IDn�����Ȓ֗Y�"""
    # ,��goproject_service�cf�����Ȓ֗
    # project = project_service.get(db, id=project_id)
    # if not project:
    #     raise HTTPException(
    #         status_code=404,
    #         detail="Project not found"
    #     )
    # if not project.is_public and project.user_id != current_user["id"]:
    #     raise HTTPException(
    #         status_code=403,
    #         detail="Not enough permissions"
    #     )
    
    # �ï�����Y
    return {
        "id": str(project_id),
        "name": "����������",
        "description": "����������n�",
        "is_public": True,
        "user_id": current_user["id"],
        "created_at": "2025-04-18T12:00:00",
        "updated_at": "2025-04-18T12:00:00"
    }


@router.put("/{project_id}", response_model=Project)
def update_project(
    *,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
    project_id: UUID,
    project_in: ProjectUpdate,
) -> Any:
    """�����Ȓ��Y�"""
    # ,��goproject_service�cf�����Ȓ֗���
    # project = project_service.get(db, id=project_id)
    # if not project:
    #     raise HTTPException(
    #         status_code=404,
    #         detail="Project not found"
    #     )
    # if project.user_id != current_user["id"]:
    #     raise HTTPException(
    #         status_code=403,
    #         detail="Not enough permissions"
    #     )
    # project = project_service.update(db, db_obj=project, obj_in=project_in)
    
    # �ï�����Y
    return {
        "id": str(project_id),
        "name": project_in.name if project_in.name else "����������",
        "description": project_in.description if project_in.description else "��U�_������n�",
        "is_public": project_in.is_public if project_in.is_public is not None else True,
        "user_id": current_user["id"],
        "created_at": "2025-04-18T12:00:00",
        "updated_at": "2025-04-18T13:30:00"
    }


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project(
    *,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
    project_id: UUID,
) -> Any:
    """�����ȒJdY�"""
    # ,��goproject_service�cf�����Ȓ֗�Jd
    # project = project_service.get(db, id=project_id)
    # if not project:
    #     raise HTTPException(
    #         status_code=404,
    #         detail="Project not found"
    #     )
    # if project.user_id != current_user["id"]:
    #     raise HTTPException(
    #         status_code=403,
    #         detail="Not enough permissions"
    #     )
    # project = project_service.remove(db, id=project_id)
    
    # Jd�BoU��UjD204 No Content	
    return None
