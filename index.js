/**
 * SmartKhet — Notification Service
 * ==================================
 * Kafka consumer that dispatches advisory notifications
 * via Push (FCM), WhatsApp, SMS (MSG91), and IVR (Twilio).
 *
 * Topics consumed:
 *   notification.outbound   — generic notification events
 *   advisory.events         — daily advisory triggers
 *   disease.events          — disease analysis completed
 *
 * Author: Axora / SmartKhet Team
 */

const { Kafka } = require("kafkajs");
const axios = require("axios");
const admin = require("firebase-admin");
const twilio = require("twilio");
const express = require("express");
const Redis = require("ioredis");

const app = express();
app.use(express.json());

// ── Config ─────────────────────────────────────────────────────────────────────

const {
  KAFKA_BOOTSTRAP_SERVERS = "localhost:9092",
  REDIS_URL = "redis://localhost:6379",
  MSG91_API_KEY = "",
  MSG91_SENDER_ID = "SKHETT",
  META_WHATSAPP_TOKEN = "",
  META_PHONE_NUMBER_ID = "",
  TWILIO_ACCOUNT_SID = "",
  TWILIO_AUTH_TOKEN = "",
  TWILIO_FROM_NUMBER = "",
  FCM_SERVICE_ACCOUNT_PATH = "/secrets/firebase-service-account.json",
  PORT = 8006,
} = process.env;

const TOPICS = {
  NOTIFICATION_OUTBOUND: "notification.outbound",
  ADVISORY_EVENTS: "advisory.events",
  DISEASE_EVENTS: "disease.events",
};

// ── Clients ────────────────────────────────────────────────────────────────────

// Firebase Admin (FCM push)
let fcmApp = null;
try {
  const serviceAccount = require(FCM_SERVICE_ACCOUNT_PATH);
  fcmApp = admin.initializeApp({
    credential: admin.credential.cert(serviceAccount),
  });
  console.log("FCM initialised ✅");
} catch (e) {
  console.warn("FCM not initialised (service account not found):", e.message);
}

// Twilio (IVR + SMS backup)
const twilioClient = TWILIO_ACCOUNT_SID
  ? twilio(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
  : null;

// Redis (dedup + rate limiting)
const redis = new Redis(REDIS_URL);

// WhatsApp API base
const WA_API = `https://graph.facebook.com/v19.0/${META_PHONE_NUMBER_ID}/messages`;


// ── Kafka Consumer Setup ───────────────────────────────────────────────────────

const kafka = new Kafka({
  clientId: "smartkhet-notification-service",
  brokers: KAFKA_BOOTSTRAP_SERVERS.split(","),
  retry: {
    initialRetryTime: 300,
    retries: 10,
    factor: 2,
  },
});

const consumer = kafka.consumer({
  groupId: "notification-service-group",
  sessionTimeout: 30000,
  heartbeatInterval: 3000,
});

async function startKafkaConsumer() {
  await consumer.connect();
  await consumer.subscribe({
    topics: Object.values(TOPICS),
    fromBeginning: false,
  });

  await consumer.run({
    eachMessage: async ({ topic, partition, message }) => {
      let event;
      try {
        event = JSON.parse(message.value.toString());
      } catch {
        console.error("Failed to parse Kafka message");
        return;
      }

      console.log(`[Kafka] ${topic} | ${event.event ?? "unknown"}`);

      try {
        await routeNotification(topic, event);
      } catch (err) {
        console.error(`[Kafka] Handler error for ${topic}:`, err.message);
      }
    },
  });

  console.log("Kafka consumer running ✅");
}


// ── Notification Router ────────────────────────────────────────────────────────

async function routeNotification(topic, event) {
  switch (topic) {
    case TOPICS.DISEASE_EVENTS:
      if (event.event === "disease.analysis.completed") {
        await handleDiseaseNotification(event);
      }
      break;

    case TOPICS.ADVISORY_EVENTS:
      if (event.event === "advisory.daily.ready") {
        await handleDailyAdvisoryNotification(event);
      }
      break;

    case TOPICS.NOTIFICATION_OUTBOUND:
      await handleDirectNotification(event);
      break;

    default:
      console.warn(`No handler for topic: ${topic}`);
  }
}

// ── Disease Notification ──────────────────────────────────────────────────────

async function handleDiseaseNotification(event) {
  const { farmer_id, disease, analysis_id } = event;
  if (!farmer_id) return;

  // Dedup — don't notify twice for same analysis
  const dedupKey = `notif_sent:disease:${analysis_id}`;
  const alreadySent = await redis.get(dedupKey);
  if (alreadySent) return;

  const farmer = await fetchFarmerContact(farmer_id);
  if (!farmer) return;

  const message =
    `🔬 *SmartKhet रोग विश्लेषण पूरा*\n\n` +
    `फ़सल में *${formatDisease(disease)}* पाया गया।\n` +
    `ऐप खोलें और उपचार की जानकारी देखें।\n\n` +
    `विश्लेषण ID: ${analysis_id?.slice(0, 8)}`;

  await dispatchNotification({
    farmer,
    message,
    push_title: "रोग विश्लेषण तैयार",
    push_body: `${formatDisease(disease)} पाया गया — उपचार देखें`,
    push_data: { type: "disease_result", analysis_id, screen: "DiseaseResult" },
  });

  await redis.setex(dedupKey, 86400, "1");
}

// ── Daily Advisory Notification ───────────────────────────────────────────────

async function handleDailyAdvisoryNotification(event) {
  const { farmer_id, advisory_count, top_advisory } = event;
  if (!farmer_id) return;

  // Rate limit: max 1 daily advisory per farmer per day
  const rateLimitKey = `daily_advisory:${farmer_id}:${new Date().toISOString().slice(0, 10)}`;
  const alreadySent = await redis.get(rateLimitKey);
  if (alreadySent) return;

  const farmer = await fetchFarmerContact(farmer_id);
  if (!farmer) return;

  const pushBody = top_advisory?.message ?? "आज की कृषि सलाह तैयार है";

  await dispatchNotification({
    farmer,
    message: `🌾 *आज की SmartKhet सलाह*\n\n${pushBody}\n\nपूरी जानकारी के लिए ऐप खोलें।`,
    push_title: "आज की खेती सलाह ☀️",
    push_body: pushBody.slice(0, 100),
    push_data: { type: "daily_advisory", screen: "Home" },
  });

  await redis.setex(rateLimitKey, 86400, "1");
}

// ── Direct / Generic Notification ─────────────────────────────────────────────

async function handleDirectNotification(event) {
  const { farmer_id, channels, message, push_title, push_body, push_data } = event;
  if (!farmer_id || !message) return;

  const farmer = await fetchFarmerContact(farmer_id);
  if (!farmer) return;

  await dispatchNotification({
    farmer,
    message,
    push_title: push_title ?? "SmartKhet",
    push_body: push_body ?? message.slice(0, 100),
    push_data: push_data ?? {},
    force_channels: channels,
  });
}

// ── Dispatch Engine ───────────────────────────────────────────────────────────

async function dispatchNotification({ farmer, message, push_title, push_body, push_data, force_channels }) {
  const channels = force_channels ?? determineChannels(farmer);
  const results = [];

  for (const channel of channels) {
    try {
      switch (channel) {
        case "push":
          if (farmer.fcm_token) {
            await sendFCMPush(farmer.fcm_token, push_title, push_body, push_data);
            results.push({ channel: "push", status: "sent" });
          }
          break;

        case "whatsapp":
          if (farmer.phone && META_WHATSAPP_TOKEN) {
            await sendWhatsAppMessage(farmer.phone, message);
            results.push({ channel: "whatsapp", status: "sent" });
          }
          break;

        case "sms":
          if (farmer.phone && MSG91_API_KEY) {
            await sendSMS(farmer.phone, stripMarkdown(message));
            results.push({ channel: "sms", status: "sent" });
          }
          break;

        case "ivr":
          if (farmer.phone && twilioClient && farmer.preferred_ivr) {
            await initiateIVRCall(farmer.phone, push_body);
            results.push({ channel: "ivr", status: "initiated" });
          }
          break;
      }
    } catch (err) {
      console.error(`[${channel}] Send failed for farmer ${farmer.id}:`, err.message);
      results.push({ channel, status: "failed", error: err.message });
    }
  }

  // Log delivery to DB
  await logDelivery(farmer.id, channels, results, message);
  return results;
}

function determineChannels(farmer) {
  // Priority: Push → WhatsApp → SMS
  // IVR only for farmers who explicitly opted in
  const channels = [];
  if (farmer.fcm_token) channels.push("push");
  if (farmer.uses_whatsapp) channels.push("whatsapp");
  if (!farmer.fcm_token && !farmer.uses_whatsapp) channels.push("sms"); // fallback
  if (farmer.preferred_ivr) channels.push("ivr");
  return channels.length ? channels : ["sms"];
}


// ── Channel Implementations ───────────────────────────────────────────────────

async function sendFCMPush(token, title, body, data = {}) {
  if (!fcmApp) throw new Error("FCM not initialised");

  await admin.messaging().send({
    token,
    notification: { title, body },
    data: Object.fromEntries(
      Object.entries(data).map(([k, v]) => [k, String(v)])
    ),
    android: {
      priority: "high",
      notification: {
        channelId: "smartkhet_advisory",
        sound: "default",
        clickAction: "FLUTTER_NOTIFICATION_CLICK",
      },
    },
  });
}

async function sendWhatsAppMessage(phone, message) {
  await axios.post(
    WA_API,
    {
      messaging_product: "whatsapp",
      to: `91${phone}`,
      type: "text",
      text: { body: message, preview_url: false },
    },
    { headers: { Authorization: `Bearer ${META_WHATSAPP_TOKEN}` } }
  );
}

async function sendSMS(phone, message) {
  // MSG91 DLT-compliant transactional SMS
  await axios.post(
    "https://api.msg91.com/api/v5/flow/",
    {
      template_id: "smartkhet_advisory_v1",
      short_url: 0,
      mobiles: `91${phone}`,
      authkey: MSG91_API_KEY,
      message: message.slice(0, 160),  // 1 SMS unit
      sender: MSG91_SENDER_ID,
    }
  );
}

async function initiateIVRCall(phone, message) {
  if (!twilioClient) throw new Error("Twilio not initialised");

  // Twilio TTS call — reads advisory in Hindi
  const twiml = `
    <Response>
      <Say language="hi-IN" voice="Polly.Aditi">
        नमस्ते! यह SmartKhet की ओर से आपकी कृषि सलाह है।
        ${message}
        दोबारा सुनने के लिए 1 दबाएँ।
      </Say>
      <Gather numDigits="1">
        <Say language="hi-IN">1 दबाएँ दोबारा सुनने के लिए।</Say>
      </Gather>
    </Response>
  `;

  await twilioClient.calls.create({
    to: `+91${phone}`,
    from: TWILIO_FROM_NUMBER,
    twiml,
  });
}

// ── Utility Functions ─────────────────────────────────────────────────────────

async function fetchFarmerContact(farmerId) {
  const cacheKey = `farmer_contact:${farmerId}`;
  const cached = await redis.get(cacheKey);
  if (cached) return JSON.parse(cached);

  try {
    const resp = await axios.get(
      `http://farmer-service:8001/farmers/${farmerId}/profile`,
      { headers: { "X-Internal-Key": process.env.INTERNAL_API_KEY ?? "" } }
    );
    const farmer = {
      id: resp.data.id,
      phone: resp.data.phone,
      fcm_token: resp.data.fcm_token ?? null,
      uses_whatsapp: resp.data.uses_whatsapp ?? true,
      preferred_ivr: resp.data.preferred_ivr ?? false,
      language: resp.data.preferred_language ?? "hi",
    };
    await redis.setex(cacheKey, 300, JSON.stringify(farmer));
    return farmer;
  } catch (e) {
    console.error(`Could not fetch farmer ${farmerId}:`, e.message);
    return null;
  }
}

async function logDelivery(farmerId, channels, results, message) {
  try {
    await axios.post("http://farmer-service:8001/farmers/internal/log-delivery", {
      farmer_id: farmerId,
      channels,
      results,
      message_preview: message.slice(0, 100),
    }, { headers: { "X-Internal-Key": process.env.INTERNAL_API_KEY ?? "" } });
  } catch { /* Non-critical — don't rethrow */ }
}

function formatDisease(label) {
  if (!label) return "अज्ञात रोग";
  const parts = label.split("___");
  const condition = parts[1] ?? label;
  return condition.replace(/_/g, " ");
}

function stripMarkdown(text) {
  return text.replace(/\*([^*]+)\*/g, "$1").replace(/\n+/g, " ").trim();
}


// ── Health Check ──────────────────────────────────────────────────────────────

app.get("/health", (req, res) => {
  res.json({ status: "healthy", service: "notification-service", version: "1.0.0" });
});

// ── Graceful Shutdown ─────────────────────────────────────────────────────────

async function shutdown() {
  console.log("Shutting down notification service...");
  await consumer.disconnect();
  await redis.quit();
  process.exit(0);
}
process.on("SIGTERM", shutdown);
process.on("SIGINT", shutdown);


// ── Startup ───────────────────────────────────────────────────────────────────

app.listen(PORT, () => {
  console.log(`Notification Service HTTP on port ${PORT}`);
});

startKafkaConsumer().catch((err) => {
  console.error("Kafka consumer failed to start:", err);
  process.exit(1);
});
