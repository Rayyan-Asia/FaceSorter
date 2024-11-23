fetch('../face_batches_find.json')
  .then(response => response.json())
  .then(data => {
    let currentIndex = 0;

    const mainImage = document.getElementById('main-image');
    const seeMoreButton = document.getElementById('see-more');
    const prevButton = document.getElementById('prev');
    const nextButton = document.getElementById('next');

    const updateImage = () => {
      const currentEntry = data[currentIndex];
      mainImage.src = currentEntry.crop_batch[0];
    };

    updateImage();

    prevButton.addEventListener('click', () => {
      currentIndex = (currentIndex - 1 + data.length) % data.length;
      updateImage();
    });

    nextButton.addEventListener('click', () => {
      currentIndex = (currentIndex + 1) % data.length;
      updateImage();
    });

    seeMoreButton.addEventListener('click', () => {
      const currentEntry = data[currentIndex];
      localStorage.setItem('batchImages', JSON.stringify(currentEntry.batch));
      window.location.href = 'detail.html';
    });
  })
  .catch(err => console.error('Failed to load JSON:', err));
