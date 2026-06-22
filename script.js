const tabs = Array.from(document.querySelectorAll('.tab'));
const panels = Array.from(document.querySelectorAll('.analysis-figure'));

tabs.forEach((tab) => {
  tab.addEventListener('click', () => {
    const target = tab.dataset.figure;
    tabs.forEach((item) => {
      const active = item === tab;
      item.classList.toggle('is-active', active);
      item.setAttribute('aria-selected', active ? 'true' : 'false');
    });
    panels.forEach((panel) => {
      panel.classList.toggle('is-hidden', panel.dataset.panel !== target);
    });
  });
});

const copyButton = document.querySelector('[data-copy-target]');
if (copyButton) {
  copyButton.addEventListener('click', async () => {
    const target = document.getElementById(copyButton.dataset.copyTarget);
    const text = target ? target.innerText.trim() : '';
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      const original = copyButton.textContent;
      copyButton.textContent = 'Copied';
      window.setTimeout(() => {
        copyButton.textContent = original;
      }, 1400);
    } catch (error) {
      copyButton.textContent = 'Select BibTeX';
    }
  });
}
