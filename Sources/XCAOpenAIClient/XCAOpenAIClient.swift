import Foundation
import OpenAPIRuntime
import OpenAPIURLSession

public struct OpenAIClient {
    
    public let client: Client
    private let urlSession = URLSession.shared
    private let apiKey: String
    
    // Add an instance of ElevenLabsClient for text-to-speech with Eleven Labs.
    private let elevenLabsClient: ElevenLabsClient
    
    // Default to using Eleven Labs for text-to-speech unless overridden.
    public var useElevenLabsForTTS: Bool = true
    
    public init(apiKey: String, elevenLabsApiKey: String) {
        self.client = Client(
            serverURL: try! Servers.server1(),
            transport: URLSessionTransport(),
            middlewares: [AuthMiddleware(apiKey: apiKey)]
        )
        self.elevenLabsClient = ElevenLabsClient(apiKey: elevenLabsApiKey)
        self.apiKey = apiKey
    }
    
    // MARK: - ChatGPT Prompt
    
    public func promptChatGPT(
        prompt: String,
        model: Components.Schemas.CreateChatCompletionRequest.modelPayload.Value2Payload = .gpt_hyphen_4,
        assistantPrompt: String = "You are a an AI-based clone of a real person named Will Dzierson. Here is your biography based on his information: I am a technologist with a strong background in user experience, healthcare AI, and product design. I grew up in Albany, NY, and attended Emerson College. In 2005, I moved to San Francisco, where I spent 15 years leading large-scale projects for companies like Google, Salesforce, and Grand Rounds, including work on Google Search and other mobile applications. I returned to Boston in 2019 and now reside in Saratoga Springs, NY. My professional focus has been at the intersection of healthcare and AI, where I’ve led the development of innovative products like a personal health record app designed to help patients manage their health data using AI-driven natural language processing. I’ve also built applications that integrate AI tools into user-friendly platforms, such as my recent project, Noodle AI, aimed at centralizing patient health management via WhatsApp and other mobile-first technologies. In addition to my work in technology, I have interests in freelance consulting, mentoring in healthcare AI, and exploring ways to improve patient care through digital solutions. I am deeply passionate about helping underserved communities and continue to explore opportunities to apply AI to healthcare challenges globally. >> Tone of voice note: please keep the tone lightheared and avoid speaking in a monotone fashion.",
        prevMessages: [Components.Schemas.ChatCompletionRequestMessage] = []) async throws -> String {
        
        let response = try await client.createChatCompletion(body: .json(.init(
            messages: [.ChatCompletionRequestAssistantMessage(.init(content: assistantPrompt, role: .assistant))]
            + prevMessages
            + [.ChatCompletionRequestUserMessage(.init(content: .case1(prompt), role: .user))],
            model: .init(value1: nil, value2: model))))
        
        switch response {
        case .ok(let body):
            let json = try body.body.json
            guard let content = json.choices.first?.message.content else {
                throw "No Response"
            }
            return content
        case .undocumented(let statusCode, let payload):
            throw "OpenAIClientError - statuscode: \(statusCode), \(payload)"
        }
    }
    
    // MARK: - Eleven Labs Text-to-Speech
    
    public func generateSpeechFromElevenLabs(text: String, voiceId: String, completion: @escaping (Result<Data, Error>) -> Void) {
        elevenLabsClient.generateSpeechFrom(text: text, voiceId: voiceId, completion: completion)
    }
    
    // MARK: - OpenAI Text-to-Speech (Fallback)
    
    public func generateSpeechFromOpenAI(input: String,
                                         model: Components.Schemas.CreateSpeechRequest.modelPayload.Value2Payload = .tts_hyphen_1,
                                         voice: Components.Schemas.CreateSpeechRequest.voicePayload = .alloy,
                                         format: Components.Schemas.CreateSpeechRequest.response_formatPayload = .aac) async throws -> Data {
        
        let response = try await client.createSpeech(body: .json(
            .init(
                model: .init(value1: nil, value2: model),
                input: input,
                voice: voice,
                response_format: format
            )))
        
        switch response {
        case .ok(let response):
            switch response.body {
            case .any(let body):
                var data = Data()
                for try await byte in body {
                    data.append(contentsOf: byte)
                }
                return data
            }
        case .undocumented(let statusCode, let payload):
            throw "OpenAIClientError - statuscode: \(statusCode), \(payload)"
        }
    }
    
    // Function to handle speech generation based on which TTS service is selected (Eleven Labs by default).
    public func generateSpeech(text: String, voiceId: String? = nil) async throws -> Data {
        if useElevenLabsForTTS, let voiceId = voiceId {
            // Use Eleven Labs for text-to-speech if enabled
            return try await withCheckedThrowingContinuation { continuation in
                elevenLabsClient.generateSpeechFrom(text: text, voiceId: voiceId) { result in
                    switch result {
                    case .success(let data):
                        continuation.resume(returning: data)
                    case .failure(let error):
                        continuation.resume(throwing: error)
                    }
                }
            }
        } else {
            // Use OpenAI's TTS if Eleven Labs is not used or voiceId is nil
            return try await generateSpeechFromOpenAI(input: text)
        }
    }
    
    // MARK: - OpenAI Whisper API (Transcription)
    
    public func generateAudioTransciptions(audioData: Data, fileName: String = "recording.m4a") async throws -> String {
        var request = URLRequest(url: URL(string: "https://api.openai.com/v1/audio/transcriptions")!)
        let boundary: String = UUID().uuidString
        request.timeoutInterval = 30
        request.httpMethod = "POST"
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        
        let bodyBuilder = MultipartFormDataBodyBuilder(boundary: boundary, entries: [
            .file(paramName: "file", fileName: fileName, fileData: audioData, contentType: "audio/mpeg"),
            .string(paramName: "model", value: "whisper-1"),
            .string(paramName: "response_format", value: "text")
        ])
        request.httpBody = bodyBuilder.build()
        let (data, resp) = try await urlSession.data(for: request)
        guard let httpResp = resp as? HTTPURLResponse, httpResp.statusCode == 200 else {
            throw "Invalid Status Code \((resp as? HTTPURLResponse)?.statusCode ?? -1)"
        }
        guard let text = String(data: data, encoding: .utf8) else {
            throw "Invalid format"
        }
        
        return text
    }
}
