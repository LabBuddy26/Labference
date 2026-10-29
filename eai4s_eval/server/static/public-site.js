(() => {
  const toggle = document.querySelector('.nav-toggle');
  const nav = document.querySelector('.site-nav');
  if (toggle && nav) {
    const setMenuOpen = (open) => {
      nav.classList.toggle('open', open);
      toggle.setAttribute('aria-expanded', String(open));
    };
    toggle.addEventListener('click', () => {
      setMenuOpen(!nav.classList.contains('open'));
    });
    nav.addEventListener('click', (event) => {
      if (event.target.closest('a')) setMenuOpen(false);
    });
    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape' && nav.classList.contains('open')) {
        setMenuOpen(false);
        toggle.focus();
      }
    });
    document.addEventListener('pointerdown', (event) => {
      if (!nav.classList.contains('open') || nav.contains(event.target) || toggle.contains(event.target)) return;
      setMenuOpen(false);
    });
  }

  const scrollBtn = document.getElementById('scrollBtn');
  if (scrollBtn) {
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const onScroll = () => scrollBtn.classList.toggle('visible', window.scrollY > 320);
    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();
    scrollBtn.addEventListener('click', () => window.scrollTo({ top: 0, behavior: reducedMotion ? 'auto' : 'smooth' }));
  }

  const demoButtons = [...document.querySelectorAll('.demo-tab')];
  if (demoButtons.length) {
    const selectDemo = (filter) => {
      demoButtons.forEach((button) => {
        const active = button.dataset.filter === filter;
        button.classList.toggle('active', active);
        button.setAttribute('aria-pressed', String(active));
      });
      document.querySelectorAll('[data-demo-card]').forEach((card) => {
        const hidden = filter !== 'all' && card.dataset.track !== filter;
        card.hidden = hidden;
        if (hidden) card.querySelector('video')?.pause();
      });
    };
    demoButtons.forEach((button) => button.addEventListener('click', () => selectDemo(button.dataset.filter)));
    selectDemo('all');
  }

  const copyButton = document.getElementById('copyBib');
  const bibtex = document.getElementById('bibtex');
  const copyStatus = document.getElementById('copyBibStatus');
  if (copyButton && bibtex) {
    const fallbackSelect = () => {
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(bibtex);
      selection?.removeAllRanges();
      selection?.addRange(range);
    };
    copyButton.addEventListener('click', async () => {
      const value = bibtex.textContent.trim();
      let copied = false;
      try {
        if (navigator.clipboard?.writeText) {
          await navigator.clipboard.writeText(value);
          copied = true;
        } else {
          fallbackSelect();
          copied = document.execCommand('copy');
        }
      } catch (_) {
        fallbackSelect();
      }
      copyButton.textContent = copied ? 'Copied!' : 'Select text';
      if (copyStatus) copyStatus.textContent = copied
        ? 'Citation copied to the clipboard.'
        : 'Citation selected. Copy it with your keyboard or context menu.';
      window.setTimeout(() => { copyButton.textContent = 'Copy'; }, 1800);
    });
  }

  document.querySelectorAll('.card-media img, .detail-media img').forEach((image) => {
    const markMissing = () => {
      image.hidden = true;
      image.parentElement?.classList.add('missing-media');
    };
    image.addEventListener('error', markMissing);
    if (image.complete && image.naturalWidth === 0) markMissing();
  });

  document.querySelectorAll('[data-hover-video]').forEach((video) => {
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    // Task cards have a full-card navigation link layered above the media, so
    // listen on the card itself rather than requiring the pointer to hit the
    // underlying video element.
    const hoverTarget = video.closest('.task-card') || video;
    const source = video.querySelector('source[data-src]');
    const ensureSource = () => {
      if (!source || source.src) return;
      source.src = source.dataset.src;
      video.load();
    };
    const play = () => {
      ensureSource();
      const promise = video.play();
      if (promise && typeof promise.catch === 'function') promise.catch(() => {});
    };
    const pause = () => {
      video.pause();
      if (Number.isFinite(video.duration)) video.currentTime = 0;
    };
    if (!reducedMotion) {
      hoverTarget.addEventListener('pointerenter', play);
      hoverTarget.addEventListener('pointerleave', pause);
      video.addEventListener('focus', play);
    }
    video.addEventListener('blur', pause);
    video.addEventListener('click', () => (video.paused ? play() : pause()));
    video.addEventListener('keydown', (event) => {
      if (event.key !== 'Enter' && event.key !== ' ') return;
      event.preventDefault();
      video.paused ? play() : pause();
    });
    video.addEventListener('playing', () => video.parentElement?.classList.add('is-playing'));
    video.addEventListener('pause', () => video.parentElement?.classList.remove('is-playing'));
    video.addEventListener('error', () => video.parentElement?.classList.add('video-unavailable'));
  });

  // Detail pages show both camera views together; muted autoplay is allowed by
  // browsers and keeps the two perspectives synchronized on page load.
  document.querySelectorAll('[data-autoplay-video]').forEach((video) => {
    const promise = video.play();
    if (promise && typeof promise.catch === 'function') promise.catch(() => {});
  });

  const taskGrid = document.querySelector('[data-task-grid]');
  if (taskGrid) {
    const cards = [...taskGrid.querySelectorAll('[data-task-card]')];
    const search = document.querySelector('[data-task-search]');
    const track = document.querySelector('[data-task-track]');
    const status = document.querySelector('[data-task-status]');
    const action = document.querySelector('[data-task-action]');
    const count = document.querySelector('[data-result-count]');
    const params = new URLSearchParams(window.location.search);
    if (track && params.has('track')) track.value = params.get('track');
    if (status && params.has('status')) status.value = params.get('status');
    if (action && params.has('action')) action.value = params.get('action');
    const update = () => {
      const query = String(search?.value || '').trim().toLowerCase();
      let visible = 0;
      cards.forEach((card) => {
        const matches = (!query || card.dataset.search.includes(query))
          && (!track?.value || card.dataset.track === track.value)
          && (!status?.value || card.dataset.status === status.value)
          && (!action?.value || card.dataset.actions.split(',').includes(action.value));
        card.hidden = !matches;
        if (matches) visible += 1;
      });
      if (count) count.textContent = `${visible} tasks`;
    };
    [search, track, status, action].forEach((control) => control?.addEventListener('input', update));
    update();
  }

  const objectGrid = document.querySelector('[data-object-grid]');
  if (objectGrid) {
    const cards = [...objectGrid.querySelectorAll('[data-object-card]')];
    const search = document.querySelector('[data-object-search]');
    const category = document.querySelector('[data-object-category]');
    const status = document.querySelector('[data-object-status]');
    const count = document.querySelector('[data-object-result-count]');
    const params = new URLSearchParams(window.location.search);
    if (category && params.has('category')) category.value = params.get('category');
    if (status && params.has('status')) status.value = params.get('status');
    const update = () => {
      const query = String(search?.value || '').trim().toLowerCase();
      let visible = 0;
      cards.forEach((card) => {
        const matches = (!query || card.dataset.search.includes(query))
          && (!category?.value || card.dataset.category === category.value)
          && (!status?.value || card.dataset.status === status.value);
        card.hidden = !matches;
        if (matches) visible += 1;
      });
      if (count) count.textContent = `${visible} objects`;
    };
    [search, category, status].forEach((control) => control?.addEventListener('input', update));
    update();
  }

  const splitButtons = [...document.querySelectorAll('[data-split-button]')];
  if (splitButtons.length) {
    const selectSplit = (split) => {
      splitButtons.forEach((button) => {
        const active = button.dataset.splitButton === split;
        button.classList.toggle('active', active);
        button.setAttribute('aria-selected', String(active));
      });
      document.querySelectorAll('[data-leaderboard-panel]').forEach((panel) => {
        panel.hidden = panel.dataset.leaderboardPanel !== split;
      });
      try { window.localStorage.setItem('labbench-leaderboard-split', split); } catch (_) {}
    };
    splitButtons.forEach((button) => button.addEventListener('click', () => selectSplit(button.dataset.splitButton)));
    let saved = '';
    try { saved = window.localStorage.getItem('labbench-leaderboard-split') || ''; } catch (_) {}
    selectSplit(splitButtons.some((button) => button.dataset.splitButton === saved) ? saved : 'sim_randomized');
  }

  document.querySelectorAll('[data-leaderboard-detail-toggle]').forEach((button) => {
    button.addEventListener('click', () => {
      const row = document.getElementById(button.dataset.leaderboardDetailToggle);
      if (!row) return;
      const expanded = row.hidden;
      row.hidden = !expanded;
      button.setAttribute('aria-expanded', String(expanded));
      button.textContent = expanded ? 'Close' : 'Details';
    });
  });

  const docsNav = document.querySelector('.docs-nav');
  if (docsNav) {
    const sections = [...docsNav.querySelectorAll('a')]
      .map((link) => link.getAttribute('href'))
      .filter((href) => href?.startsWith('#'))
      .map((href) => document.querySelector(href))
      .filter(Boolean);
    const observer = new IntersectionObserver((entries) => {
      const visible = entries
        .filter((entry) => entry.isIntersecting)
        .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)
        .at(-1);
      if (!visible) return;
      docsNav.querySelectorAll('a.active').forEach((link) => link.classList.remove('active'));
      docsNav.querySelector(`a[href="#${visible.target.id}"]`)?.classList.add('active');
    }, { rootMargin: '-40% 0px -55% 0px', threshold: 0 });
    sections.forEach((section) => observer.observe(section));
  }

})();
