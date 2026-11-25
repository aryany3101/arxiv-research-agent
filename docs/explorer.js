const exampleLinks = [...document.querySelectorAll('[data-example]')];
const answers = [...document.querySelectorAll('.answer')];

function showExample(hash, focusAnswer = false) {
  const answer = answers.find((item) => `#${item.id}` === hash);
  if (!answer) return;
  for (const item of answers) item.hidden = item !== answer;
  for (const link of exampleLinks) {
    if (link.hash === hash) link.setAttribute('aria-current', 'true');
    else link.removeAttribute('aria-current');
  }
  if (focusAnswer) answer.focus();
}

const initialExample = exampleLinks.find((link) => link.hash === location.hash);
showExample(initialExample?.hash ?? exampleLinks[0].hash);
window.addEventListener('hashchange', () => showExample(location.hash || exampleLinks[0].hash));

for (const link of exampleLinks) {
  link.addEventListener('click', (event) => {
    if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    if (location.hash !== link.hash) history.pushState(null, '', link.hash);
    showExample(link.hash, true);
  });
}
