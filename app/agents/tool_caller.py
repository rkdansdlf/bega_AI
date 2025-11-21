"""
도구 호출을 관리하는 시스템입니다.

이 모듈은 LLM이 실제 도구들을 안전하게 호출할 수 있도록 
인터페이스를 제공하고 실행 결과를 관리합니다.
"""

import logging
from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ToolCall:
    """도구 호출 요청을 나타내는 클래스"""
    tool_name: str
    parameters: Dict[str, Any]
    
    def __str__(self):
        return f"ToolCall(tool_name='{self.tool_name}', parameters={self.parameters})"

    def to_dict(self) -> Dict[str, Any]:
        """JSON serialization을 위한 딕셔너리 변환"""
        return {
            "tool_name": self.tool_name,
            "parameters": self.parameters
        }

@dataclass 
class ToolResult:
    """도구 실행 결과를 나타내는 클래스"""
    success: bool
    data: Any
    message: str
    
    def __str__(self):
        return f"ToolResult(success={self.success}, message='{self.message}')"
    
    def to_dict(self) -> Dict[str, Any]:
        """JSON serialization을 위한 딕셔너리 변환"""
        return {
            "success": self.success,
            "data": self.data,
            "message": self.message
        }

class ToolCaller:
    """
    도구 호출을 관리하는 클래스
    
    이 클래스는 다음과 같은 역할을 합니다:
    1. 사용 가능한 도구들을 등록하고 관리
    2. 도구 호출 요청을 안전하게 실행
    3. 실행 결과를 표준화된 형태로 반환
    4. 오류 처리 및 로깅
    """
    
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}
        
    def register_tool(
        self, 
        tool_name: str, 
        description: str, 
        parameters_schema: Dict[str, str],
        function: Callable
    ) -> None:
        """
        새로운 도구를 등록합니다.
        
        Args:
            tool_name: 도구 이름
            description: 도구 설명
            parameters_schema: 매개변수 스키마 {param_name: description}
            function: 실제 실행할 함수
        """
        self.tools[tool_name] = {
            "description": description,
            "parameters_schema": parameters_schema,
            "function": function
        }
        logger.debug(f"[ToolCaller] Registered tool: {tool_name}")
        
    def get_tool_descriptions(self) -> str:
        """등록된 모든 도구들의 설명을 반환합니다."""
        descriptions = []
        for tool_name, tool_info in self.tools.items():
            descriptions.append(f"**{tool_name}**: {tool_info['description']}")
            
            # 매개변수 정보 추가
            if tool_info['parameters_schema']:
                param_lines = []
                for param_name, param_desc in tool_info['parameters_schema'].items():
                    param_lines.append(f"  - {param_name}: {param_desc}")
                descriptions.append("\n".join(param_lines))
            descriptions.append("")  # 빈 줄 추가
            
        return "\n".join(descriptions)
    
    def execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """
        도구 호출을 실행합니다.
        
        Args:
            tool_call: 실행할 도구 호출 정보
            
        Returns:
            도구 실행 결과
        """
        logger.info(f"[ToolCaller] Executing: {tool_call}")
        
        # 도구 존재 여부 확인
        if tool_call.tool_name not in self.tools:
            error_msg = f"존재하지 않는 도구입니다: {tool_call.tool_name}"
            logger.error(f"[ToolCaller] {error_msg}")
            return ToolResult(
                success=False,
                data={},
                message=error_msg
            )
        
        tool_info = self.tools[tool_call.tool_name]
        tool_function = tool_info["function"]
        
        try:
            # 매개변수 유효성 검사
            required_params = set(tool_info["parameters_schema"].keys())
            provided_params = set(tool_call.parameters.keys())
            
            # 필수 매개변수 확인 (일부는 선택적일 수 있으므로 경고만)
            missing_params = required_params - provided_params
            if missing_params:
                logger.warning(f"[ToolCaller] Missing parameters: {missing_params}")
            
            # 도구 함수 실행
            result = tool_function(**tool_call.parameters)
            
            # 결과 타입 확인
            if isinstance(result, ToolResult):
                logger.info(f"[ToolCaller] Tool executed successfully: {tool_call.tool_name}")
                return result
            else:
                # 일반 함수 결과를 ToolResult로 래핑
                logger.info(f"[ToolCaller] Tool executed, wrapping result: {tool_call.tool_name}")
                return ToolResult(
                    success=True,
                    data=result,
                    message=f"{tool_call.tool_name} 실행 완료"
                )
                
        except TypeError as e:
            error_msg = f"매개변수 오류: {e}"
            logger.error(f"[ToolCaller] Parameter error for {tool_call.tool_name}: {e}")
            return ToolResult(
                success=False,
                data={},
                message=error_msg
            )
        except Exception as e:
            error_msg = f"도구 실행 중 오류: {e}"
            logger.error(f"[ToolCaller] Execution error for {tool_call.tool_name}: {e}")
            return ToolResult(
                success=False,
                data={},
                message=error_msg
            )
    
    def execute_multiple_tools(self, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """
        여러 도구를 순차적으로 실행합니다.
        
        Args:
            tool_calls: 실행할 도구 호출 목록
            
        Returns:
            각 도구의 실행 결과 목록
        """
        logger.info(f"[ToolCaller] Executing {len(tool_calls)} tools")
        
        results = []
        for i, tool_call in enumerate(tool_calls):
            logger.info(f"[ToolCaller] Executing tool {i+1}/{len(tool_calls)}: {tool_call.tool_name}")
            result = self.execute_tool(tool_call)
            results.append(result)
            
            # 실패 시 로깅 (계속 진행)
            if not result.success:
                logger.warning(f"[ToolCaller] Tool {tool_call.tool_name} failed: {result.message}")
        
        return results
    
    def list_available_tools(self) -> List[str]:
        """사용 가능한 도구 목록을 반환합니다."""
        return list(self.tools.keys())
    
    def get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """특정 도구의 스키마 정보를 반환합니다."""
        if tool_name not in self.tools:
            return None
        
        return {
            "name": tool_name,
            "description": self.tools[tool_name]["description"],
            "parameters_schema": self.tools[tool_name]["parameters_schema"]
        }