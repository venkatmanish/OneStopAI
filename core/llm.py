from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx

from core.settings import get_settings


class LLMClient:
    groq_timeout = 6
    ollama_timeout = 30

    def __init__(self) -> None:
        self.settings = get_settings()

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.1,
        model: str | None = None,
        timeout: float | None = None,
    ) -> str:
        if self.settings.groq_api_key:
            try:
                return self._groq(prompt, system, temperature, model=model, timeout=timeout)
            except Exception as first_error:
                if model and model != self.settings.groq_model:
                    try:
                        return self._groq(
                            prompt,
                            system,
                            temperature,
                            model=self.settings.groq_model,
                            timeout=timeout,
                        )
                    except Exception:
                        pass
                fallback = self._ollama(prompt, system, temperature)
                if fallback.startswith("LLM backend is not configured"):
                    return f"{fallback} Groq error: {first_error}"
                return fallback
        return self._ollama(prompt, system, temperature)

    def json_complete(self, prompt: str, system: str | None = None) -> dict[str, Any]:
        text = self.complete(prompt, system=system, temperature=0.0)
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
        return {}

    def stream_complete(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.1,
        model: str | None = None,
        timeout: float | None = None,
    ) -> Iterator[str]:
        if self.settings.groq_api_key:
            yielded = False
            try:
                for chunk in self._groq_stream(
                    prompt,
                    system,
                    temperature,
                    model=model,
                    timeout=timeout,
                ):
                    yielded = True
                    yield chunk
                return
            except Exception as first_error:
                if yielded:
                    yield f"\n\nResponse stream interrupted: {first_error}"
                    return
                if model and model != self.settings.groq_model:
                    try:
                        for chunk in self._groq_stream(
                            prompt,
                            system,
                            temperature,
                            model=self.settings.groq_model,
                            timeout=timeout,
                        ):
                            yield chunk
                        return
                    except Exception:
                        pass
                fallback_text = self.complete(prompt, system=system, temperature=temperature)
                if fallback_text.startswith("LLM backend is not configured"):
                    yield f"{fallback_text} Groq error: {first_error}"
                else:
                    yield fallback_text
                return

        yield from self._ollama_stream(prompt, system, temperature)

    def _groq(
        self,
        prompt: str,
        system: str | None,
        temperature: float,
        model: str | None = None,
        timeout: float | None = None,
    ) -> str:
        from groq import Groq

        client = Groq(
            api_key=self.settings.groq_api_key,
            timeout=timeout or self.groq_timeout,
            max_retries=0,
        )
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(
            model=model or self.settings.groq_model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    def _groq_stream(
        self,
        prompt: str,
        system: str | None,
        temperature: float,
        model: str | None = None,
        timeout: float | None = None,
    ) -> Iterator[str]:
        from groq import Groq

        client = Groq(
            api_key=self.settings.groq_api_key,
            timeout=timeout or self.groq_timeout,
            max_retries=0,
        )
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        stream = client.chat.completions.create(
            model=model or self.settings.groq_model,
            messages=messages,
            temperature=temperature,
            stream=True,
        )
        for event in stream:
            if not event.choices:
                continue
            content = event.choices[0].delta.content
            if content:
                yield content

    def _ollama(self, prompt: str, system: str | None, temperature: float) -> str:
        payload = {
            "model": self.settings.ollama_model,
            "prompt": prompt if not system else f"{system}\n\n{prompt}",
            "stream": False,
            "options": {"temperature": temperature},
        }
        try:
            response = httpx.post(
                f"{self.settings.ollama_base_url.rstrip('/')}/api/generate",
                json=payload,
                timeout=self.ollama_timeout,
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as exc:
            return (
                "LLM backend is not configured. Set GROQ_API_KEY or run Ollama locally. "
                f"Backend error: {exc}"
            )

    def _ollama_stream(
        self,
        prompt: str,
        system: str | None,
        temperature: float,
    ) -> Iterator[str]:
        payload = {
            "model": self.settings.ollama_model,
            "prompt": prompt if not system else f"{system}\n\n{prompt}",
            "stream": True,
            "options": {"temperature": temperature},
        }
        try:
            with httpx.stream(
                "POST",
                f"{self.settings.ollama_base_url.rstrip('/')}/api/generate",
                json=payload,
                timeout=self.ollama_timeout,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    token = data.get("response")
                    if token:
                        yield token
                    if data.get("done"):
                        break
        except Exception as exc:
            yield (
                "LLM backend is not configured. Set GROQ_API_KEY or run Ollama locally. "
                f"Backend error: {exc}"
            )
