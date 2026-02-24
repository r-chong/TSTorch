/**
 * Manual test script for TSTorch Dataset Visualizer
 * Run this in the browser console at http://localhost:5173/
 */

console.log('=== TSTorch Dataset Visualizer Test ===\n');

// Test 1: Check if page loaded with 6 dataset cards
const cards = document.querySelectorAll('.card');
console.log(`✓ Test 1: Dataset cards loaded: ${cards.length} cards found`);
if (cards.length !== 6) {
  console.error(`❌ EXPECTED 6 cards, got ${cards.length}`);
} else {
  console.log('  ✓ Correct number of cards (6)');
}

// Test 2: List all dataset names
const cardTitles = Array.from(document.querySelectorAll('.card-title')).map(el => el.textContent);
console.log(`\n✓ Test 2: Dataset names: ${cardTitles.join(', ')}`);

// Test 3: Check initial button states
const btnTrain = document.getElementById('btn-train');
const btnPlay = document.getElementById('btn-play');
const btnPause = document.getElementById('btn-pause');

console.log('\n✓ Test 3: Initial button states:');
console.log(`  - Train All: ${btnTrain.disabled ? 'disabled' : 'enabled'}`);
console.log(`  - Play All: ${btnPlay.disabled ? 'disabled' : 'enabled'}`);
console.log(`  - Pause All: ${btnPause.disabled ? 'disabled' : 'enabled'}`);

// Test 4: Check for any console errors so far
console.log('\n✓ Test 4: Console errors check');
console.log('  (Check above for any red error messages)');

// Test 5: Click Train All button
console.log('\n✓ Test 5: Clicking "Train All" button...');
const startTime = Date.now();

// Add event listener to track when training completes
const originalText = btnTrain.textContent;
const observer = new MutationObserver((mutations) => {
  mutations.forEach((mutation) => {
    if (mutation.type === 'childList' || mutation.type === 'characterData') {
      const currentText = btnTrain.textContent;
      if (currentText === 'Retrain All') {
        const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
        console.log(`\n✓ Training completed in ${elapsed} seconds!`);
        console.log('  - Button text changed to "Retrain All"');
        console.log('  - Play/Pause buttons should now be enabled');
        console.log(`  - Play All enabled: ${!btnPlay.disabled}`);
        console.log(`  - Pause All enabled: ${!btnPause.disabled}`);
        
        // Check if animations started
        setTimeout(() => {
          const firstCanvas = document.querySelector('canvas');
          console.log('\n✓ Test 6: Animation check');
          console.log('  - Canvas found:', !!firstCanvas);
          console.log('  - Check visually if decision boundaries are animating');
          
          // Final summary
          console.log('\n=== TEST SUMMARY ===');
          console.log('✓ Page loaded successfully');
          console.log(`✓ ${cards.length} dataset cards rendered`);
          console.log('✓ Training completed');
          console.log('✓ Animations should be playing');
          console.log('\nManually verify:');
          console.log('  1. No red errors in console');
          console.log('  2. Decision boundaries are animating on all 6 cards');
          console.log('  3. Stats show epoch/loss/accuracy for each card');
        }, 1000);
        
        observer.disconnect();
      }
    }
  });
});

observer.observe(btnTrain, {
  childList: true,
  characterData: true,
  subtree: true
});

// Trigger the training
btnTrain.click();
console.log('  - Training started, waiting for completion...');
console.log('  - This will take approximately 20-30 seconds');
