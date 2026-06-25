const puppeteer = require('puppeteer');
const path = require('path');

const wait = ms => new Promise(r => setTimeout(r, ms));
const outDir = __dirname;
const BASE = 'http://localhost:5000';

(async () => {
  const browser = await puppeteer.launch({
    headless: true,
    executablePath: 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 900 });

  // =========== 1. LANDING — INTRO ===========
  await page.goto(BASE, { waitUntil: 'networkidle0' });
  await wait(1000);
  await page.screenshot({ path: path.join(outDir, '01-landing-intro.png'), fullPage: true });
  console.log('1/7 Landing intro captured');

  // =========== 2. LANDING PAGE ===========
  // Intro transitions at 3700ms + 780ms fade
  await wait(3500);
  await page.screenshot({ path: path.join(outDir, '02-landing-page.png'), fullPage: true });
  console.log('2/7 Landing page captured');

  // =========== 3. LOGIN PAGE ===========
  await page.goto(BASE + '/login', { waitUntil: 'networkidle0' });
  await wait(600);
  await page.screenshot({ path: path.join(outDir, '03-login.png'), fullPage: true });
  console.log('3/7 Login page captured');

  // =========== 4. REGISTER PAGE ===========
  await page.goto(BASE + '/register', { waitUntil: 'networkidle0' });
  await wait(600);
  await page.screenshot({ path: path.join(outDir, '04-register.png'), fullPage: true });
  console.log('4/7 Register page captured');

  // =========== 5. FORGOT PASSWORD ===========
  await page.goto(BASE + '/forgot', { waitUntil: 'networkidle0' });
  await wait(600);
  await page.screenshot({ path: path.join(outDir, '05-forgot-password.png'), fullPage: true });
  console.log('5/7 Forgot password captured');

  // =========== 6. REGISTER & LOGIN (get session) ===========
  const username = 'demouser_' + Date.now();
  const email = username + '@demo.com';
  const password = 'demo123456';

  await page.goto(BASE + '/register', { waitUntil: 'networkidle0' });
  await wait(500);

  // Fill register form
  await page.type('#name', 'Demo User', { delay: 15 });
  await page.type('#username', username, { delay: 10 });
  await page.type('#email', email, { delay: 10 });
  await page.type('#password', password, { delay: 10 });
  await page.type('#confirm', password, { delay: 10 });
  await page.click('#registerBtn');

  // Wait for redirect to /predict (after successful registration)
  await page.waitForFunction(() => window.location.pathname === '/predict', { timeout: 10000 });
  await wait(1500);

  // =========== 6. PREDICT DASHBOARD ===========
  await page.screenshot({ path: path.join(outDir, '06-predict-dashboard.png'), fullPage: true });
  console.log('6/7 Predict dashboard captured');

  // =========== 7. PREDICT — WITH RESULT ===========
  // Click a sample chip (Fantasy)
  await page.click('#surpriseBtn');
  await wait(300);
  await page.click('.chip[data-s="fantasy"]');
  await wait(300);
  // Click Predict
  await page.click('#predictBtn');
  await wait(3000);
  await page.screenshot({ path: path.join(outDir, '07-predict-result.png'), fullPage: true });
  console.log('7/7 Predict with result captured');

  await browser.close();
  console.log('\nAll 7 screenshots saved to screenshots/');
})();
