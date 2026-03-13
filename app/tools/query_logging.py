from __future__ import annotations

import logging

DOCUMENT_QUERY_COMPONENT = "DocumentQueryTool"
REGULATION_QUERY_COMPONENT = "RegulationQuery"

ACTION_SEARCH_DOCUMENTS = "search_documents"
ACTION_SEARCH_REGULATION = "search_regulation"
ACTION_GET_REGULATION_BY_CATEGORY = "get_regulation_by_category"
ACTION_FIND_RELATED_REGULATIONS = "find_related_regulations"
ACTION_VALIDATE_REGULATION_REFERENCE = "validate_regulation_reference"


def log_query_start(
    logger: logging.Logger,
    *,
    component: str,
    action: str,
    value: str,
) -> None:
    logger.info(
        "[%s] event=query_start action=%s value=%s",
        component,
        action,
        value,
    )


def log_query_success(
    logger: logging.Logger,
    *,
    component: str,
    action: str,
    count: int,
    detail: str | None = None,
) -> None:
    if detail:
        logger.info(
            "[%s] event=query_success action=%s count=%d detail=%s",
            component,
            action,
            count,
            detail,
        )
        return
    logger.info(
        "[%s] event=query_success action=%s count=%d",
        component,
        action,
        count,
    )


def log_query_empty(
    logger: logging.Logger,
    *,
    component: str,
    action: str,
    value: str,
) -> None:
    logger.warning(
        "[%s] event=query_empty action=%s value=%s",
        component,
        action,
        value,
    )


def log_query_retry(
    logger: logging.Logger,
    *,
    component: str,
    action: str,
    reason: str,
) -> None:
    logger.warning(
        "[%s] event=query_retry action=%s reason=%s",
        component,
        action,
        reason,
    )


def log_dependency_missing(
    logger: logging.Logger,
    *,
    component: str,
    action: str,
    dependency: str,
) -> None:
    logger.warning(
        "[%s] event=dependency_missing action=%s dependency=%s",
        component,
        action,
        dependency,
    )


def log_query_error(
    logger: logging.Logger,
    *,
    component: str,
    action: str,
    error: str,
) -> None:
    logger.error(
        "[%s] event=query_error action=%s error=%s",
        component,
        action,
        error,
    )


def build_retry_warning_message(component: str, action: str) -> str:
    return f"[{component}] event=query_retry action={action} reason=%s"
