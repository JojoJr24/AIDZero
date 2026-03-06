package com.aidzero.android

import android.app.Application
import android.content.Context
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DrawerValue
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalDrawerSheet
import androidx.compose.material3.ModalNavigationDrawer
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.rememberDrawerState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Menu
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

data class ChatMessage(
    val role: MessageRole,
    val text: String,
)

enum class MessageRole {
    USER,
    ASSISTANT,
    SYSTEM,
}

data class AndroidUiState(
    val coreUrl: String = "",
    val status: String = "Enter the core IP running on the LAN.",
    val connected: Boolean = false,
    val loading: Boolean = false,
)

class AndroidChatViewModel(application: Application) : AndroidViewModel(application) {
    private val prefs = application.getSharedPreferences("aidzero_android", Context.MODE_PRIVATE)
    private val _uiState = MutableStateFlow(
        AndroidUiState(
            coreUrl = prefs.getString(KEY_CORE_URL, "") ?: "",
            status = "Enter the core IP running on the LAN.",
        )
    )
    val uiState: StateFlow<AndroidUiState> = _uiState.asStateFlow()
    val messages = mutableStateListOf(
        ChatMessage(
            role = MessageRole.SYSTEM,
            text = "Same LAN only. Set the core IP, check the connection, and start chatting.",
        )
    )

    fun updateCoreUrl(value: String) {
        _uiState.value = _uiState.value.copy(coreUrl = value)
    }

    fun checkConnection() {
        val rawUrl = requireCoreUrl() ?: return
        persistCoreUrl(rawUrl)
        _uiState.value = _uiState.value.copy(loading = true, status = "Contacting core...")
        viewModelScope.launch {
            try {
                val health = withContext(Dispatchers.IO) { CoreApiClient(rawUrl).health() }
                _uiState.value = _uiState.value.copy(
                    connected = true,
                    status = "Connected to ${health.coreUrl} (${health.agent})",
                )
            } catch (error: Exception) {
                messages += ChatMessage(MessageRole.SYSTEM, error.message ?: "Unknown Android app error.")
                _uiState.value = _uiState.value.copy(
                    connected = false,
                    status = error.message ?: "Unknown Android app error.",
                )
            } finally {
                _uiState.value = _uiState.value.copy(loading = false)
            }
        }
    }

    fun sendPrompt(prompt: String) {
        val cleanPrompt = prompt.trim()
        if (cleanPrompt.isEmpty()) {
            _uiState.value = _uiState.value.copy(status = "Write a prompt before sending.")
            return
        }
        val rawUrl = requireCoreUrl() ?: return
        persistCoreUrl(rawUrl)
        messages += ChatMessage(MessageRole.USER, cleanPrompt)
        _uiState.value = _uiState.value.copy(loading = true, status = "Contacting core...")
        viewModelScope.launch {
            try {
                val result = withContext(Dispatchers.IO) {
                    val client = CoreApiClient(rawUrl)
                    client.addPrompt(cleanPrompt)
                    client.runPrompt(cleanPrompt) to client.baseUrl
                }
                val response = result.first
                val baseUrl = result.second
                messages += ChatMessage(MessageRole.ASSISTANT, response)
                _uiState.value = _uiState.value.copy(
                    connected = true,
                    status = "Response received from $baseUrl",
                )
            } catch (error: Exception) {
                messages += ChatMessage(MessageRole.SYSTEM, error.message ?: "Unknown Android app error.")
                _uiState.value = _uiState.value.copy(
                    connected = false,
                    status = error.message ?: "Unknown Android app error.",
                )
            } finally {
                _uiState.value = _uiState.value.copy(loading = false)
            }
        }
    }

    fun resetSession() {
        val rawUrl = requireCoreUrl() ?: return
        persistCoreUrl(rawUrl)
        _uiState.value = _uiState.value.copy(loading = true, status = "Contacting core...")
        viewModelScope.launch {
            try {
                val result = withContext(Dispatchers.IO) {
                    val client = CoreApiClient(rawUrl)
                    client.resetSession() to client.baseUrl
                }
                val response = result.first
                val baseUrl = result.second
                messages += ChatMessage(MessageRole.SYSTEM, response)
                _uiState.value = _uiState.value.copy(
                    connected = true,
                    status = "Session reset on $baseUrl",
                )
            } catch (error: Exception) {
                messages += ChatMessage(MessageRole.SYSTEM, error.message ?: "Unknown Android app error.")
                _uiState.value = _uiState.value.copy(
                    connected = false,
                    status = error.message ?: "Unknown Android app error.",
                )
            } finally {
                _uiState.value = _uiState.value.copy(loading = false)
            }
        }
    }

    private fun requireCoreUrl(): String? {
        val rawUrl = _uiState.value.coreUrl.trim()
        if (rawUrl.isEmpty()) {
            _uiState.value = _uiState.value.copy(status = "Set the core IP first.")
            return null
        }
        return rawUrl
    }

    private fun persistCoreUrl(rawUrl: String) {
        prefs.edit().putString(KEY_CORE_URL, rawUrl).apply()
    }

    companion object {
        private const val KEY_CORE_URL = "core_url"
    }
}

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            AndroidClientTheme {
                Surface(modifier = Modifier.fillMaxSize(), color = Color(0xFF07101E)) {
                    AndroidClientApp()
                }
            }
        }
    }
}

@Composable
private fun AndroidClientApp(viewModel: AndroidChatViewModel = viewModel()) {
    val state by viewModel.uiState.collectAsStateWithLifecycle()
    AndroidClientContent(
        state = state,
        messages = viewModel.messages,
        onUpdateCoreUrl = viewModel::updateCoreUrl,
        onCheckConnection = viewModel::checkConnection,
        onResetSession = viewModel::resetSession,
        onSendPrompt = viewModel::sendPrompt,
    )
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun AndroidClientContent(
    state: AndroidUiState,
    messages: List<ChatMessage>,
    onUpdateCoreUrl: (String) -> Unit,
    onCheckConnection: () -> Unit,
    onResetSession: () -> Unit,
    onSendPrompt: (String) -> Unit,
) {
    var prompt by remember { mutableStateOf("") }
    val drawerState = rememberDrawerState(initialValue = DrawerValue.Closed)
    val scope = rememberCoroutineScope()

    ModalNavigationDrawer(
        drawerState = drawerState,
        drawerContent = {
            ModalDrawerSheet(
                drawerContainerColor = Color(0xFF0F172A),
                drawerContentColor = Color.White,
            ) {
                DrawerContent(
                    coreUrl = state.coreUrl,
                    loading = state.loading,
                    onUpdateCoreUrl = onUpdateCoreUrl,
                    onCheckConnection = {
                        scope.launch { drawerState.close() }
                        onCheckConnection()
                    },
                    onResetSession = {
                        scope.launch { drawerState.close() }
                        onResetSession()
                    },
                )
            }
        },
    ) {
        Scaffold(
            modifier = Modifier
                .fillMaxSize()
                .background(
                    Brush.verticalGradient(
                        colors = listOf(Color(0xFF020617), Color(0xFF09111F)),
                    )
                )
                .safeDrawingPadding(),
            containerColor = Color.Transparent,
        ) { padding ->
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(padding)
                    .padding(horizontal = 12.dp, vertical = 10.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                HeaderBar(
                    status = state.status,
                    connected = state.connected,
                    loading = state.loading,
                    onOpenMenu = {
                        scope.launch { drawerState.open() }
                    },
                )

                Card(
                    modifier = Modifier.weight(1f),
                    colors = CardDefaults.cardColors(containerColor = Color(0xE60F172A)),
                    shape = RoundedCornerShape(22.dp),
                ) {
                    Column(
                        modifier = Modifier
                            .fillMaxSize()
                            .padding(12.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        LazyColumn(
                            modifier = Modifier.weight(1f),
                            verticalArrangement = Arrangement.spacedBy(8.dp),
                        ) {
                            items(messages) { message ->
                                MessageBubble(message)
                            }
                        }

                        OutlinedTextField(
                            value = prompt,
                            onValueChange = { prompt = it },
                            modifier = Modifier.fillMaxWidth(),
                            label = { Text("Prompt") },
                            placeholder = { Text("Write your message") },
                            minLines = 2,
                            colors = darkFieldColors(),
                        )

                        Button(
                            onClick = {
                                onSendPrompt(prompt)
                                prompt = ""
                            },
                            modifier = Modifier.fillMaxWidth(),
                            enabled = !state.loading,
                        ) {
                            Text("Send", color = Color.White)
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun HeaderBar(
    status: String,
    connected: Boolean,
    loading: Boolean,
    onOpenMenu: () -> Unit,
) {
    Card(
        colors = CardDefaults.cardColors(containerColor = Color(0xD9091424)),
        shape = RoundedCornerShape(22.dp),
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 14.dp, vertical = 12.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
                    Text(
                        text = "LOCAL ANDROID CLIENT",
                        color = Color(0xFF38BDF8),
                        style = MaterialTheme.typography.labelSmall,
                        fontWeight = FontWeight.Bold,
                    )
                    Text(
                        text = "AIDZero",
                        color = Color.White,
                        style = MaterialTheme.typography.headlineSmall,
                        fontWeight = FontWeight.Black,
                    )
                }
                IconButton(onClick = onOpenMenu) {
                    Icon(
                        imageVector = Icons.Filled.Menu,
                        contentDescription = "Open menu",
                        tint = Color.White,
                    )
                }
            }

            StatusChip(
                text = status,
                connected = connected,
                loading = loading,
            )
        }
    }
}

@Composable
private fun DrawerContent(
    coreUrl: String,
    loading: Boolean,
    onUpdateCoreUrl: (String) -> Unit,
    onCheckConnection: () -> Unit,
    onResetSession: () -> Unit,
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 12.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        Text(
            text = "Connection",
            color = Color.White,
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.Bold,
        )
        Text(
            text = "Configure the core endpoint and session actions.",
            color = Color(0xFF94A3B8),
            style = MaterialTheme.typography.bodySmall,
        )
        OutlinedTextField(
            value = coreUrl,
            onValueChange = onUpdateCoreUrl,
            modifier = Modifier.fillMaxWidth(),
            label = { Text("Core IP or URL") },
            placeholder = { Text("192.168.1.20:8765") },
            singleLine = true,
            colors = darkFieldColors(),
        )
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(
                onClick = onCheckConnection,
                modifier = Modifier.fillMaxWidth(),
                enabled = !loading,
            ) {
                Text("Check", color = Color.White)
            }
            Button(
                onClick = onResetSession,
                modifier = Modifier.fillMaxWidth(),
                enabled = !loading,
                colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFEA580C)),
            ) {
                Text("Reset", color = Color.White)
            }
        }
    }
}

@Composable
private fun StatusChip(text: String, connected: Boolean, loading: Boolean) {
    val background = when {
        loading -> Color(0x3322C55E)
        connected -> Color(0x3322C55E)
        else -> Color(0x33F97316)
    }
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .background(background, RoundedCornerShape(999.dp))
            .padding(horizontal = 12.dp, vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        if (loading) {
            CircularProgressIndicator(
                modifier = Modifier.height(16.dp),
                strokeWidth = 2.dp,
                color = Color(0xFF38BDF8),
            )
        }
        Text(
            text = text,
            color = Color.White,
            style = MaterialTheme.typography.bodyMedium,
        )
    }
}

@Composable
private fun MessageBubble(message: ChatMessage) {
    val container = when (message.role) {
        MessageRole.USER -> Color(0x3328A7F7)
        MessageRole.ASSISTANT -> Color(0xFF111C31)
        MessageRole.SYSTEM -> Color(0x33F97316)
    }
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .background(container, RoundedCornerShape(18.dp))
            .padding(12.dp),
    ) {
        Text(
            text = message.text,
            color = Color.White,
            style = MaterialTheme.typography.bodyMedium,
        )
    }
}

@Composable
private fun AndroidClientTheme(content: @Composable () -> Unit) {
    MaterialTheme(content = content)
}

@Composable
private fun darkFieldColors() = OutlinedTextFieldDefaults.colors(
    focusedTextColor = Color.White,
    unfocusedTextColor = Color.White,
    focusedLabelColor = Color.White,
    unfocusedLabelColor = Color(0xFFE2E8F0),
    focusedPlaceholderColor = Color(0xFFCBD5E1),
    unfocusedPlaceholderColor = Color(0xFF94A3B8),
    cursorColor = Color(0xFF38BDF8),
    focusedBorderColor = Color(0xFF38BDF8),
    unfocusedBorderColor = Color(0xFF475569),
)

@Preview(showBackground = true)
@Composable
private fun MessageBubblePreview() {
    AndroidClientTheme {
        Column(
            modifier = Modifier
                .background(Color(0xFF07101E))
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            MessageBubble(ChatMessage(MessageRole.USER, "Hello! This is a user message."))
            MessageBubble(ChatMessage(MessageRole.ASSISTANT, "Hello! I am your assistant. How can I help?"))
            MessageBubble(ChatMessage(MessageRole.SYSTEM, "System message: Connected to core successfully."))
        }
    }
}

@Preview(showBackground = true)
@Composable
private fun StatusChipPreview() {
    AndroidClientTheme {
        Column(
            modifier = Modifier
                .background(Color(0xFF07101E))
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            StatusChip("Connected to 192.168.1.20", connected = true, loading = false)
            StatusChip("Contacting core...", connected = false, loading = true)
            StatusChip("Set the core IP first.", connected = false, loading = false)
        }
    }
}

@Preview(showBackground = true)
@Composable
private fun HeaderBarPreview() {
    AndroidClientTheme {
        Box(
            modifier = Modifier
                .background(Color(0xFF07101E))
                .padding(16.dp)
        ) {
            HeaderBar(
                status = "Connected to 192.168.1.20",
                connected = true,
                loading = false,
                onOpenMenu = {},
            )
        }
    }
}

@Preview(showSystemUi = true)
@Composable
private fun FullAppPreview() {
    AndroidClientTheme {
        AndroidClientContent(
            state = AndroidUiState(
                coreUrl = "192.168.1.10",
                status = "Connected to core",
                connected = true
            ),
            messages = listOf(
                ChatMessage(MessageRole.SYSTEM, "Same LAN only. Set the core IP, check the connection, and start chatting."),
                ChatMessage(MessageRole.USER, "Show me the project status."),
                ChatMessage(MessageRole.ASSISTANT, "The project is currently in the development phase. All core modules are active.")
            ),
            onUpdateCoreUrl = {},
            onCheckConnection = {},
            onResetSession = {},
            onSendPrompt = {}
        )
    }
}
