      function copyToClipboard(text) {
        if (navigator.clipboard && window.isSecureContext) {
          return navigator.clipboard.writeText(text);
        }
        return new Promise((resolve, reject) => {
          const textarea = document.createElement("textarea");
          textarea.value = text;
          textarea.setAttribute("readonly", "");
          textarea.style.position = "fixed";
          textarea.style.left = "-9999px";
          document.body.appendChild(textarea);
          textarea.select();
          const copied = document.execCommand("copy");
          document.body.removeChild(textarea);
          copied ? resolve() : reject(new Error("copy command failed"));
        });
      }

      document.querySelectorAll(".prompt-panel").forEach((panel) => {
        const copyButton = panel.querySelector(".copy-button");
        if (!copyButton) return; // 无复制按钮的面板（如说明面板）跳过
        const copyStatus = panel.querySelector(".copy-status");
        const promptText = panel.querySelector(".prompt-code");
        copyButton.addEventListener("click", async (event) => {
          event.preventDefault();
          copyStatus.textContent = "";
          try {
            await copyToClipboard(promptText.textContent.trim());
            copyStatus.textContent = "已复制";
          } catch {
            copyStatus.textContent = "复制失败";
          }
        });
      });
