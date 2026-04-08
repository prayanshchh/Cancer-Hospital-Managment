const root = document.documentElement;
const toggle = document.getElementById("theme-toggle");
const fileInput = document.getElementById("image-input");
const modalTriggers = document.querySelectorAll("[data-modal-target]");
const modalCloses = document.querySelectorAll("[data-modal-close]");
const storageKey = "pathofusion-theme";

const savedTheme = localStorage.getItem(storageKey);
if (savedTheme === "dark" || savedTheme === "light") {
  root.setAttribute("data-theme", savedTheme);
}

if (toggle) {
  toggle.addEventListener("click", () => {
    const current = root.getAttribute("data-theme") === "dark" ? "dark" : "light";
    const next = current === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", next);
    localStorage.setItem(storageKey, next);
  });
}

if (fileInput) {
  fileInput.addEventListener("change", (event) => {
    const [file] = event.target.files || [];
    if (!file || !file.type.startsWith("image/")) {
      return;
    }

    const reader = new FileReader();
    reader.onload = () => {
      const existingImage = document.querySelector('img[alt="Uploaded histopathology image"]');
      const visualEmpty = document.querySelector(".visual-grid .visual-card:first-child .visual-empty");

      if (existingImage) {
        existingImage.src = reader.result;
        return;
      }

      if (visualEmpty) {
        const image = document.createElement("img");
        image.alt = "Uploaded histopathology image";
        image.src = reader.result;
        visualEmpty.replaceWith(image);
      }
    };
    reader.readAsDataURL(file);
  });
}

modalTriggers.forEach((trigger) => {
  trigger.addEventListener("click", () => {
    const modal = document.getElementById(trigger.dataset.modalTarget);
    if (modal) {
      modal.classList.add("modal-overlay--open");
    }
  });
});

modalCloses.forEach((button) => {
  button.addEventListener("click", () => {
    const modal = button.closest(".modal-overlay");
    if (modal) {
      modal.classList.remove("modal-overlay--open");
    }
  });
});

document.querySelectorAll(".modal-overlay").forEach((modal) => {
  modal.addEventListener("click", (event) => {
    if (event.target === modal) {
      modal.classList.remove("modal-overlay--open");
    }
  });
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    document.querySelectorAll(".modal-overlay--open").forEach((modal) => {
      modal.classList.remove("modal-overlay--open");
    });
  }
});
