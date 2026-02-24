#!/usr/bin/env node

/**
 * Simple verification script for the TSTorch Dataset Visualizer
 * This script will open the page and provide instructions for manual verification
 */

const http = require('http');

console.log('🔍 TSTorch Dataset Visualizer - Verification Script\n');

// Check if the server is running
const options = {
  hostname: 'localhost',
  port: 5173,
  path: '/',
  method: 'GET',
};

const req = http.request(options, (res) => {
  console.log('✅ Server is running at http://localhost:5173/');
  console.log(`   Status: ${res.statusCode}\n`);
  
  console.log('📋 Manual Verification Checklist:\n');
  console.log('1. ✓ Navigate to http://localhost:5173/');
  console.log('   - The page should load with 6 dataset cards');
  console.log('   - Datasets: Simple, Diag, Split, Xor, Circle, Spiral\n');
  
  console.log('2. ✓ Click "Train All" button');
  console.log('   - Progress indicators should appear on each card');
  console.log('   - Training takes approximately 20-30 seconds');
  console.log('   - Progress bars should update during training\n');
  
  console.log('3. ✓ Wait for training to complete');
  console.log('   - All 6 decision boundaries should start animating');
  console.log('   - Progress overlays should disappear');
  console.log('   - Animations should be smooth and continuous\n');
  
  console.log('4. ✓ Check browser console (F12 or Cmd+Option+I)');
  console.log('   - Look for any red error messages');
  console.log('   - There should be NO JavaScript errors\n');
  
  console.log('5. ✓ Test GIF export');
  console.log('   - Click any "GIF" button');
  console.log('   - Should trigger a download (e.g., tstorch-simple.gif)');
  console.log('   - Button should show progress percentage');
  console.log('   - No errors should appear in console\n');
  
  console.log('6. ✓ Visual quality check');
  console.log('   - Decision boundaries should be clearly visible');
  console.log('   - Colors should be distinct (blue/red for classes)');
  console.log('   - Data points should be properly rendered');
  console.log('   - Animations should be smooth (not choppy)\n');
  
  console.log('Expected Results:');
  console.log('  • All 6 datasets train successfully');
  console.log('  • All animations play smoothly');
  console.log('  • No JavaScript console errors');
  console.log('  • GIF export works without errors');
  console.log('  • Visual quality is high\n');
  
  console.log('Opening browser now...\n');
  
  // Open the browser
  const { exec } = require('child_process');
  exec('open http://localhost:5173/', (error) => {
    if (error) {
      console.error('❌ Could not open browser automatically');
      console.log('   Please open http://localhost:5173/ manually\n');
    } else {
      console.log('✅ Browser opened successfully\n');
    }
    
    console.log('Please perform the verification steps above and report back with:');
    console.log('  1. Any JavaScript console errors (if any)');
    console.log('  2. Whether all 6 datasets trained and are animating');
    console.log('  3. Whether GIF export works');
    console.log('  4. Overall visual quality assessment\n');
  });
});

req.on('error', (e) => {
  console.error('❌ Server is not running!');
  console.error(`   Error: ${e.message}\n`);
  console.log('Please start the dev server first:');
  console.log('  cd packages/demo-web');
  console.log('  npx vite\n');
  process.exit(1);
});

req.end();
