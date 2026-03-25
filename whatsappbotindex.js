/**
 * SmartKhet — WhatsApp Bot
 * =========================
 * Meta Cloud API webhook handler for WhatsApp Business.
 * Handles text queries, image uploads, and voice messages from farmers.
 *
 * Supported interactions:
 *   📝 Text  → Intent classification → Advisory
 *   📸 Image → Disease detection → Treatment advice
 *   🎤 Voice → STT → Intent → Advisory
 *   📋 Menu  → Button/List message navigation
 *
 * Author: Axora / SmartKhet Team
 */

const express = require("express");
const axios = require("axios");
const FormData = require("form-data");
const crypto = require("crypto");

const app = express();
app.use(express.json());

// ── Config ─────────────────────────────────────────────────────────────────────

const {
  META_WHATSAPP_TOKEN,
  META_VERIFY_TOKEN,
  META_PHONE_NUMBER_ID,
  API_GATEWAY_URL = "http://localhost:8000",
  INTERNAL_API_KEY = "internal-service-key",
  PORT = 3000,
} = process.env;

const WA_API_BASE = `https://graph.facebook.com/v19.0/${META_PHONE_NUMBER_ID}`;

// ── Webhook Verification ───────────────────────────────────────────────────────

app.get("/webhook", (req, res) => {
  const mode = req.query["hub.mode"];
  const token = req.query["hub.verify_token"];
  const challenge = req.query["hub.challenge"];

  if (mode === "subscribe" && token === META_VERIFY_TOKEN) {
    console.log("WhatsApp webhook verified ✅");
    return res.status(200).send(challenge);
  }
  res.sendStatus(403);
});

// ── Signature Verification Middleware ─────────────────────────────────────────

function verifySignature(req, res, next) {
  const signature = req.headers["x-hub-signature-256"];
  if (!signature) return res.sendStatus(401);

  const expected = "sha256=" + crypto
    .createHmac("sha256", META_WHATSAPP_TOKEN)
    .update(JSON.stringify(req.body))
    .digest("hex");

  if (!crypto.timingSafeEqual(Buffer.from(signature), Buffer.from(expected))) {
    return res.sendStatus(401);
  }
  next();
}

// ── Main Webhook Handler ──────────────────────────────────────────────────────

app.post("/webhook", verifySignature, async (req, res) => {
  // Acknowledge immediately — Meta requires < 5s response
  res.sendStatus(200);

  const body = req.body;
  if (body.object !== "whatsapp_business_account") return;

  for (const entry of body.entry ?? []) {
    for (const change of entry.changes ?? []) {
      const value = change.value;
      if (!value?.messages?.length) continue;

      for (const message of value.messages) {
        const from = message.from;  // Farmer's WhatsApp number
        const waId = message.id;

        // Mark as read
        await markRead(waId).catch(() => {});

        try {
          await routeMessage(message, from, value.contacts?.[0]);
        } catch (err) {
          console.error(`Error handling message from ${from}:`, err.message);
          await sendText(from,
            "माफ़ करें, अभी कोई समस्या आ रही है। थोड़ी देर बाद फिर कोशिश करें। 🙏"
          );
        }
      }
    }
  }
});

// ── Message Router ────────────────────────────────────────────────────────────

async function routeMessage(message, from, contact) {
  const farmerName = contact?.profile?.name ?? "किसान";

  switch (message.type) {
    case "text":
      await handleTextMessage(message.text.body, from, farmerName);
      break;

    case "image":
      await handleImageMessage(message.image, from, farmerName);
      break;

    case "audio":
    case "voice":
      await handleVoiceMessage(message.audio ?? message.voice, from, farmerName);
      break;

    case "interactive":
      await handleInteractiveReply(message.interactive, from, farmerName);
      break;

    case "location":
      await handleLocationMessage(message.location, from, farmerName);
      break;

    default:
      await sendText(from,
        `नमस्ते ${farmerName}! मैं SmartKhet हूँ 🌾\n\n` +
        `आप मुझे:\n• 📝 टाइप करके लिख सकते हैं\n• 📸 फसल की फ़ोटो भेज सकते हैं\n` +
        `• 🎤 आवाज़ में पूछ सकते हैं\n\nमेनू के लिए "MENU" लिखें।`
      );
  }
}

// ── Text Message Handler ──────────────────────────────────────────────────────

async function handleTextMessage(text, from, farmerName) {
  const normalised = text.trim().toLowerCase();

  // Shortcut commands
  if (["menu", "मेनू", "help", "मदद", "start", "शुरू"].includes(normalised)) {
    return sendMainMenu(from, farmerName);
  }

  if (normalised.startsWith("bhav") || normalised.startsWith("भाव")) {
    // e.g., "भाव गेहूँ गोरखपुर"
    const parts = text.split(/\s+/);
    const commodity = parts[1] ?? "wheat";
    const district = parts[2] ?? "";
    return handleMandiQuery(commodity, district, from, farmerName);
  }

  // General NLP processing
  const payload = await callNLPService(text, from, "hi");
  await deliverAdvisory(payload, from, farmerName);
}

// ── Image Handler (Disease Detection) ────────────────────────────────────────

async function handleImageMessage(image, from, farmerName) {
  await sendText(from, "फ़ोटो मिली ✅ विश्लेषण हो रहा है... 🔍 (2-3 सेकंड)");

  // Download image from Meta servers
  const imageBuffer = await downloadMedia(image.id);
  if (!imageBuffer) {
    return sendText(from, "फ़ोटो डाउनलोड नहीं हो सकी। दोबारा भेजें।");
  }

  // Forward to Disease Detection Service
  const form = new FormData();
  form.append("file", imageBuffer, {
    filename: "crop.jpg",
    contentType: image.mime_type ?? "image/jpeg",
  });
  form.append("farmer_wa_id", from);
  form.append("language", "hi");

  let result;
  try {
    const resp = await axios.post(
      `${API_GATEWAY_URL}/api/v1/disease/analyze`,
      form,
      {
        headers: {
          ...form.getHeaders(),
          "X-Internal-Key": INTERNAL_API_KEY,
          "X-Farmer-WA": from,
        },
        timeout: 30000,
      }
    );
    result = resp.data;
  } catch (err) {
    console.error("Disease API error:", err.response?.data ?? err.message);
    return sendText(from, "फ़सल का विश्लेषण नहीं हो सका। थोड़ी देर में फिर कोशिश करें।");
  }

  // Format response
  const primary = result.primary;
  const severity = result.severity;
  const severityEmoji = { low: "🟡", medium: "🟠", high: "🔴", critical: "🆘" }[severity] ?? "⚠️";

  let responseText = `*SmartKhet रोग विश्लेषण* 🌿\n\n`;
  responseText += `*फ़सल:* ${primary.crop}\n`;
  responseText += `*समस्या:* ${primary.condition.replace(/_/g, " ")}\n`;
  responseText += `*विश्वसनीयता:* ${primary.confidence_pct}\n`;
  responseText += `*गंभीरता:* ${severityEmoji} ${severity.toUpperCase()}\n\n`;

  if (result.advisory_hindi) {
    responseText += `*सलाह:*\n${result.advisory_hindi}\n\n`;
  }

  if (result.treatment_advisory?.length) {
    responseText += `*उपाय:*\n`;
    result.treatment_advisory.slice(0, 3).forEach((step, i) => {
      responseText += `${i + 1}. ${step.action}`;
      if (step.product) responseText += ` (${step.product}${step.dosage ? ` - ${step.dosage}` : ""})`;
      responseText += "\n";
    });
  }

  if (result.estimated_yield_loss_pct) {
    responseText += `\n⚠️ संभावित उत्पादन हानि: *${result.estimated_yield_loss_pct}%* तक\n`;
  }

  responseText += `\nविश्लेषण ID: \`${result.analysis_id.slice(0, 8)}\``;

  await sendText(from, responseText);

  // Follow-up interactive buttons
  await sendInteractiveButtons(from, "और जानकारी चाहिए?", [
    { id: `treatment_detail_${result.analysis_id}`, title: "विस्तृत उपचार" },
    { id: "nearby_shop", title: "नज़दीकी दुकान" },
    { id: "main_menu", title: "मुख्य मेनू" },
  ]);
}

// ── Voice Handler ─────────────────────────────────────────────────────────────

async function handleVoiceMessage(audio, from, farmerName) {
  await sendText(from, "आवाज़ सुन रहा हूँ... 🎤");

  const audioBuffer = await downloadMedia(audio.id);
  if (!audioBuffer) {
    return sendText(from, "आवाज़ रिकॉर्ड नहीं मिली। दोबारा कोशिश करें।");
  }

  const form = new FormData();
  form.append("audio", audioBuffer, {
    filename: "voice.ogg",
    contentType: audio.mime_type ?? "audio/ogg",
  });
  form.append("farmer_wa_id", from);

  try {
    const resp = await axios.post(
      `${API_GATEWAY_URL}/api/v1/nlp/voice`,
      form,
      { headers: { ...form.getHeaders(), "X-Internal-Key": INTERNAL_API_KEY }, timeout: 20000 }
    );
    const parsed = resp.data;

    // Echo transcription back
    await sendText(from, `✍️ *आपने कहा:* "${parsed.transcription}"`);

    // Route parsed query to advisory
    await deliverAdvisory(parsed, from, farmerName);
  } catch (err) {
    await sendText(from, "आवाज़ समझ नहीं आई। कृपया दोबारा या लिखकर पूछें।");
  }
}

// ── Mandi Price Handler ───────────────────────────────────────────────────────

async function handleMandiQuery(commodity, district, from, farmerName) {
  try {
    const resp = await axios.get(
      `${API_GATEWAY_URL}/api/v1/market/sell-signal/${commodity}`,
      { params: { district }, headers: { "X-Internal-Key": INTERNAL_API_KEY } }
    );
    const data = resp.data;

    const signalEmoji = {
      SELL_NOW: "🔴 अभी बेचें",
      SELL: "🟢 बेचें",
      HOLD: "⏳ रुकें",
      HOLD_FOR_MSP: "🔵 MSP का इंतज़ार करें",
    }[data.signal] ?? data.signal;

    let msg = `*SmartKhet मंडी सलाह* 📊\n\n`;
    msg += `*फ़सल:* ${commodity.toUpperCase()}\n`;
    msg += `*आज का भाव:* ₹${data.current_price}/क्विंटल\n`;
    msg += `*7 दिन बाद अनुमान:* ₹${data.predicted_price}/क्विंटल\n`;
    msg += `*MSP:* ₹${data.msp ?? "N/A"}/क्विंटल\n\n`;
    msg += `*सलाह: ${signalEmoji}*\n${data.reason}\n`;
    if (data.best_mandi) {
      msg += `\n🏪 *सबसे अच्छी मंडी:* ${data.best_mandi} — ₹${data.best_mandi_price}/क्विंटल`;
    }

    await sendText(from, msg);
  } catch {
    await sendText(from, `${commodity} का भाव अभी उपलब्ध नहीं है। बाद में कोशिश करें।`);
  }
}

// ── Interactive Reply Handler ─────────────────────────────────────────────────

async function handleInteractiveReply(interactive, from, farmerName) {
  const replyId = interactive.button_reply?.id ?? interactive.list_reply?.id ?? "";

  if (replyId === "crop_advice") {
    await sendText(from, "अपनी मिट्टी की जानकारी भेजें:\nN P K pH नमी\nउदाहरण: 90 45 40 6.5 65");
  } else if (replyId === "disease_detect") {
    await sendText(from, "फ़सल की पत्ती या तने की साफ़ फ़ोटो भेजें 📸");
  } else if (replyId === "mandi_price") {
    await sendText(from, "फ़सल और ज़िले का नाम लिखें:\nउदाहरण: भाव गेहूँ गोरखपुर");
  } else if (replyId === "weather") {
    await sendText(from, "अपना पिनकोड भेजें मौसम जानकारी के लिए:");
  } else if (replyId === "main_menu") {
    await sendMainMenu(from, farmerName);
  } else {
    await sendMainMenu(from, farmerName);
  }
}

// ── Location Handler ──────────────────────────────────────────────────────────

async function handleLocationMessage(location, from, farmerName) {
  // Store location for context-aware recommendations
  const { latitude, longitude } = location;
  await sendText(from,
    `📍 लोकेशन मिली (${latitude.toFixed(4)}, ${longitude.toFixed(4)})\n` +
    `अब आपको स्थानीय मौसम और मंडी जानकारी मिलेगी।`
  );
}

// ── Advisory Delivery ─────────────────────────────────────────────────────────

async function deliverAdvisory(nlpPayload, from, farmerName) {
  const { intent, advisory_text, language } = nlpPayload;

  if (!advisory_text) {
    await sendText(from, `${farmerName} जी, आपकी जानकारी के लिए हम काम कर रहे हैं।`);
    return;
  }

  await sendText(from, `*SmartKhet सलाह* 🌾\n\n${advisory_text}`);
}

// ── WhatsApp UI Helpers ───────────────────────────────────────────────────────

async function sendMainMenu(from, farmerName) {
  await axios.post(
    `${WA_API_BASE}/messages`,
    {
      messaging_product: "whatsapp",
      to: from,
      type: "interactive",
      interactive: {
        type: "list",
        header: { type: "text", text: `नमस्ते ${farmerName} 🙏` },
        body: { text: "SmartKhet में आपका स्वागत है!\nक्या जानना है आज?" },
        footer: { text: "Powered by SmartKhet AI" },
        action: {
          button: "मेनू देखें",
          sections: [
            {
              title: "🌾 फ़सल सलाह",
              rows: [
                { id: "crop_advice", title: "फ़सल सुझाव", description: "मिट्टी के अनुसार फ़सल" },
                { id: "fertilizer", title: "खाद सलाह", description: "NPK और खाद की मात्रा" },
                { id: "irrigation", title: "सिंचाई सलाह", description: "कब और कितना पानी" },
              ],
            },
            {
              title: "🔬 रोग / कीट",
              rows: [
                { id: "disease_detect", title: "रोग पहचान", description: "फ़ोटो से रोग जाँच" },
                { id: "pest_query", title: "कीट प्रबंधन", description: "कीट और उपाय" },
              ],
            },
            {
              title: "💰 मंडी / मौसम",
              rows: [
                { id: "mandi_price", title: "मंडी भाव", description: "आज के दाम + पूर्वानुमान" },
                { id: "weather", title: "मौसम", description: "7 दिन का पूर्वानुमान" },
              ],
            },
          ],
        },
      },
    },
    { headers: { Authorization: `Bearer ${META_WHATSAPP_TOKEN}` } }
  );
}

async function sendText(to, text) {
  await axios.post(
    `${WA_API_BASE}/messages`,
    {
      messaging_product: "whatsapp",
      to,
      type: "text",
      text: { body: text, preview_url: false },
    },
    { headers: { Authorization: `Bearer ${META_WHATSAPP_TOKEN}` } }
  );
}

async function sendInteractiveButtons(to, bodyText, buttons) {
  await axios.post(
    `${WA_API_BASE}/messages`,
    {
      messaging_product: "whatsapp",
      to,
      type: "interactive",
      interactive: {
        type: "button",
        body: { text: bodyText },
        action: {
          buttons: buttons.map((b) => ({
            type: "reply",
            reply: { id: b.id, title: b.title.slice(0, 20) },
          })),
        },
      },
    },
    { headers: { Authorization: `Bearer ${META_WHATSAPP_TOKEN}` } }
  );
}

async function markRead(messageId) {
  await axios.post(
    `${WA_API_BASE}/messages`,
    {
      messaging_product: "whatsapp",
      status: "read",
      message_id: messageId,
    },
    { headers: { Authorization: `Bearer ${META_WHATSAPP_TOKEN}` } }
  );
}

async function downloadMedia(mediaId) {
  try {
    // Get media URL
    const urlResp = await axios.get(`https://graph.facebook.com/v19.0/${mediaId}`, {
      headers: { Authorization: `Bearer ${META_WHATSAPP_TOKEN}` },
    });
    const mediaUrl = urlResp.data.url;

    // Download actual media
    const mediaResp = await axios.get(mediaUrl, {
      headers: { Authorization: `Bearer ${META_WHATSAPP_TOKEN}` },
      responseType: "arraybuffer",
    });
    return Buffer.from(mediaResp.data);
  } catch (err) {
    console.error("Media download failed:", err.message);
    return null;
  }
}

// ── NLP Service Call ──────────────────────────────────────────────────────────

async function callNLPService(text, farmerWaId, language = "hi") {
  try {
    const resp = await axios.post(
      `${API_GATEWAY_URL}/api/v1/nlp/text`,
      { text, farmer_wa_id: farmerWaId, language },
      { headers: { "X-Internal-Key": INTERNAL_API_KEY }, timeout: 10000 }
    );
    return resp.data;
  } catch (err) {
    console.error("NLP service error:", err.message);
    return { advisory_text: "अभी कनेक्शन में समस्या है। थोड़ी देर बाद कोशिश करें।" };
  }
}

// ── Health Check ──────────────────────────────────────────────────────────────

app.get("/health", (req, res) => {
  res.json({ status: "healthy", service: "whatsapp-bot", version: "1.0.0" });
});

// ── Start Server ──────────────────────────────────────────────────────────────

app.listen(PORT, () => {
  console.log(`SmartKhet WhatsApp Bot listening on port ${PORT} ✅`);
});

module.exports = app;
