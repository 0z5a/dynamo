// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP transport for the TRT-LLM context-first handoff.
//!
//! The two legs target different workers, so the transport holds two base URLs.
//! Both are supplied by the caller: `#13416` excludes EPP discovery, and the
//! selected-prefill metadata in this crate carries the prefill worker's HTTP
//! authority rather than a context endpoint, so no URL is derived here.

use std::time::Duration;

use reqwest::{Client, StatusCode, Url};
use tokio_util::sync::CancellationToken;

use crate::trtllm_context_first::{Leg, LegFailure, LegTransport, OwnedBody, PreparedLeg};

/// Sends both legs over HTTP.
pub struct ContextFirstTransport {
    client: Client,
    context_base: Url,
    generation_base: Url,
    max_response_bytes: usize,
}

impl std::fmt::Debug for ContextFirstTransport {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ContextFirstTransport")
            .field("context_base", &self.context_base)
            .field("generation_base", &self.generation_base)
            .field("max_response_bytes", &self.max_response_bytes)
            .finish_non_exhaustive()
    }
}

impl ContextFirstTransport {
    pub fn new(
        client: Client,
        context_base: Url,
        generation_base: Url,
        max_response_bytes: usize,
    ) -> Self {
        Self {
            client,
            context_base,
            generation_base,
            max_response_bytes,
        }
    }

    fn url(&self, base: &Url, path: &str, leg: Leg) -> Result<Url, LegFailure> {
        base.join(path).map_err(|error| {
            tracing::error!(%error, %base, path, %leg, "invalid leg URL");
            LegFailure::Unavailable { leg }
        })
    }
}

/// Maps a `reqwest` failure onto the dispatch taxonomy.
///
/// A read stall and a total deadline are different failures and are reported
/// differently: the total deadline is enforced by the dispatcher around the
/// whole leg, while `ReadStall` means the HTTP client's own read timeout fired
/// because the worker stopped sending. Collapsing them would make a stalled
/// engine indistinguishable from an unreachable one.
fn classify(error: &reqwest::Error, leg: Leg, reading_body: bool) -> LegFailure {
    if error.is_timeout() {
        if reading_body {
            LegFailure::ReadStall { leg }
        } else {
            LegFailure::Deadline { leg }
        }
    } else if error.is_connect() {
        LegFailure::Connect { leg }
    } else {
        LegFailure::Unavailable { leg }
    }
}

/// Reads a response body under an explicit cap.
///
/// The cap is applied while reading rather than after, so an oversized or
/// endless body cannot make the sidecar allocate without bound.
async fn read_bounded(
    mut response: reqwest::Response,
    leg: Leg,
    max_bytes: usize,
) -> Result<Vec<u8>, LegFailure> {
    let mut collected: Vec<u8> = Vec::new();
    loop {
        let chunk = response
            .chunk()
            .await
            .map_err(|error| classify(&error, leg, true))?;
        let Some(chunk) = chunk else {
            return Ok(collected);
        };
        if collected.len() + chunk.len() > max_bytes {
            return Err(LegFailure::BodyTooLarge {
                leg,
                limit: max_bytes,
            });
        }
        collected.extend_from_slice(&chunk);
    }
}

fn check_status(status: StatusCode, leg: Leg) -> Result<(), LegFailure> {
    if status.is_success() {
        Ok(())
    } else {
        Err(LegFailure::Status {
            leg,
            status: status.as_u16(),
        })
    }
}

async fn post_leg(
    client: &Client,
    url: Url,
    prepared: &PreparedLeg,
    leg: Leg,
    cancellation: CancellationToken,
    deadline: Duration,
) -> Result<reqwest::Response, LegFailure> {
    let mut request = client
        .post(url)
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .body(prepared.body.to_vec());
    // The pinned worker rejects a protected handoff field without this header,
    // so it is sent whenever the leg carries a signature.
    if let Some((name, value)) = &prepared.auth_header {
        request = request.header(*name, value);
    }
    let request = request.send();
    tokio::select! {
        () = cancellation.cancelled() => Err(LegFailure::Unavailable { leg }),
        result = tokio::time::timeout(deadline, request) => {
            result
                .map_err(|_| LegFailure::Deadline { leg })?
                .map_err(|error| classify(&error, leg, false))
        }
    }
}

#[async_trait::async_trait]
impl LegTransport for ContextFirstTransport {
    async fn send_context(
        &self,
        leg: PreparedLeg,
        cancellation: CancellationToken,
        deadline: Duration,
    ) -> Result<Vec<u8>, LegFailure> {
        let url = self.url(&self.context_base, leg.path, Leg::Context)?;
        let response = post_leg(
            &self.client,
            url,
            &leg,
            Leg::Context,
            cancellation,
            deadline,
        )
        .await?;
        check_status(response.status(), Leg::Context)?;
        read_bounded(response, Leg::Context, self.max_response_bytes).await
    }

    async fn send_generation(
        &self,
        leg: PreparedLeg,
        cancellation: CancellationToken,
        deadline: Duration,
    ) -> Result<OwnedBody, LegFailure> {
        let url = self.url(&self.generation_base, leg.path, Leg::Generation)?;
        let response = post_leg(
            &self.client,
            url,
            &leg,
            Leg::Generation,
            cancellation.clone(),
            deadline,
        )
        .await?;
        check_status(response.status(), Leg::Generation)?;
        let bytes = read_bounded(response, Leg::Generation, self.max_response_bytes).await?;
        // The body is owned by the response; dropping it cancels the token so a
        // streaming generation stream is torn down rather than orphaned.
        Ok(OwnedBody::new(bytes, cancellation))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trtllm_context_first::CHAT_COMPLETIONS_PATH;
    use std::net::{Ipv4Addr, SocketAddr};
    use std::sync::Arc;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    /// What the scripted origin server does with a request.
    #[derive(Clone, Copy)]
    enum Origin {
        /// Answer 200 with this body.
        Body(&'static str),
        /// Answer with this status and no body.
        Status(u16),
        /// Accept the request, then never answer.
        NeverRespond,
        /// Answer 200 promising 64 KiB, send a fragment, then stall.
        StallAfterHeaders,
    }

    async fn spawn_origin(behaviour: Origin) -> (SocketAddr, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else {
                    return;
                };
                tokio::spawn(async move {
                    let mut buffer = [0u8; 4096];
                    let _ = socket.read(&mut buffer).await;
                    match behaviour {
                        Origin::Body(body) => {
                            let response = format!(
                                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                                body.len(),
                                body
                            );
                            let _ = socket.write_all(response.as_bytes()).await;
                        }
                        Origin::Status(status) => {
                            let response = format!(
                                "HTTP/1.1 {status} Error\r\ncontent-length: 0\r\nconnection: close\r\n\r\n"
                            );
                            let _ = socket.write_all(response.as_bytes()).await;
                        }
                        Origin::NeverRespond => {
                            tokio::time::sleep(Duration::from_secs(30)).await;
                        }
                        Origin::StallAfterHeaders => {
                            let head = "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: 65536\r\n\r\n{\"a\"";
                            let _ = socket.write_all(head.as_bytes()).await;
                            tokio::time::sleep(Duration::from_secs(30)).await;
                        }
                    }
                });
            }
        });
        (addr, handle)
    }

    fn base(addr: SocketAddr) -> Url {
        Url::parse(&format!("http://{addr}/")).unwrap()
    }

    fn client(read_timeout: Duration) -> Client {
        Client::builder()
            .no_proxy()
            .connect_timeout(Duration::from_millis(500))
            .read_timeout(read_timeout)
            .build()
            .unwrap()
    }

    fn leg() -> PreparedLeg {
        PreparedLeg {
            path: CHAT_COMPLETIONS_PATH,
            correlation_id: 1,
            body: bytes::Bytes::from_static(b"{}"),
            auth_header: None,
        }
    }

    fn transport(addr: SocketAddr, read_timeout: Duration, cap: usize) -> ContextFirstTransport {
        ContextFirstTransport::new(client(read_timeout), base(addr), base(addr), cap)
    }

    fn runtime() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
    }

    /// A closed port is a connect failure, a silent worker is a total deadline,
    /// and a worker that stops mid-body is a read stall: three distinct
    /// classifications from three real HTTP conditions.
    #[test]
    fn real_http_conditions_map_to_distinct_failures() {
        let runtime = runtime();

        // A port with no listener: bind then drop, so the port is free.
        let closed = runtime.block_on(async {
            let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
            listener.local_addr().unwrap()
        });
        let outcome = runtime.block_on(
            transport(closed, Duration::from_millis(300), 1024).send_context(
                leg(),
                CancellationToken::new(),
                Duration::from_secs(5),
            ),
        );
        assert!(
            matches!(outcome, Err(LegFailure::Connect { leg: Leg::Context })),
            "closed port must classify as Connect, got {outcome:?}"
        );

        // A worker that accepts but never answers, under a short total deadline.
        let (addr, server) = runtime.block_on(spawn_origin(Origin::NeverRespond));
        let outcome =
            runtime.block_on(transport(addr, Duration::from_secs(30), 1024).send_context(
                leg(),
                CancellationToken::new(),
                Duration::from_millis(300),
            ));
        assert!(
            matches!(outcome, Err(LegFailure::Deadline { leg: Leg::Context })),
            "a silent worker must classify as Deadline, got {outcome:?}"
        );
        server.abort();

        // A worker that answers 200 then stops sending mid-body, with the
        // client's own read timeout firing first.
        let (addr, server) = runtime.block_on(spawn_origin(Origin::StallAfterHeaders));
        let outcome = runtime.block_on(
            transport(addr, Duration::from_millis(300), 4096).send_context(
                leg(),
                CancellationToken::new(),
                Duration::from_secs(30),
            ),
        );
        assert!(
            matches!(outcome, Err(LegFailure::ReadStall { leg: Leg::Context })),
            "a stalled body must classify as ReadStall, got {outcome:?}"
        );
        server.abort();
    }

    /// The response cap is enforced while reading, and an oversized body is a
    /// size failure rather than a transport failure.
    #[test]
    fn an_oversized_response_is_rejected_by_the_cap() {
        let runtime = runtime();
        let (addr, server) = runtime.block_on(spawn_origin(Origin::Body(
            "{\"choices\":[],\"padding\":\"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\"}",
        )));
        let outcome = runtime.block_on(transport(addr, Duration::from_secs(5), 8).send_context(
            leg(),
            CancellationToken::new(),
            Duration::from_secs(5),
        ));
        assert!(
            matches!(
                outcome,
                Err(LegFailure::BodyTooLarge {
                    leg: Leg::Context,
                    limit: 8
                })
            ),
            "expected a size rejection, got {outcome:?}"
        );
        server.abort();
    }

    /// A non-success status carries its code, so a 503 is not reported as an
    /// unreachable worker.
    #[test]
    fn a_non_success_status_reports_the_code() {
        let runtime = runtime();
        let (addr, server) = runtime.block_on(spawn_origin(Origin::Status(503)));
        let outcome = runtime.block_on(transport(addr, Duration::from_secs(5), 1024).send_context(
            leg(),
            CancellationToken::new(),
            Duration::from_secs(5),
        ));
        assert_eq!(
            outcome,
            Err(LegFailure::Status {
                leg: Leg::Context,
                status: 503
            })
        );
        server.abort();
    }

    /// A healthy response is returned verbatim, and the generation body keeps
    /// the response cancellation.
    #[test]
    fn a_healthy_response_is_returned_and_owns_its_cancellation() {
        let runtime = runtime();
        let (addr, server) =
            runtime.block_on(spawn_origin(Origin::Body("{\"choices\":[{\"index\":0}]}")));
        let transport = transport(addr, Duration::from_secs(5), 4096);

        let context = runtime
            .block_on(transport.send_context(
                leg(),
                CancellationToken::new(),
                Duration::from_secs(5),
            ))
            .expect("context leg succeeds");
        assert_eq!(context, br#"{"choices":[{"index":0}]}"#);

        let cancellation = CancellationToken::new();
        let body = runtime
            .block_on(transport.send_generation(
                leg(),
                cancellation.clone(),
                Duration::from_secs(5),
            ))
            .expect("generation leg succeeds");
        assert_eq!(body.as_slice(), br#"{"choices":[{"index":0}]}"#);
        assert!(!body.cancellation().is_cancelled());
        cancellation.cancel();
        assert!(body.cancellation().is_cancelled());
        server.abort();
    }

    /// An already-cancelled request never reaches the worker.
    #[test]
    fn a_cancelled_request_is_not_sent() {
        let runtime = runtime();
        let (addr, server) = runtime.block_on(spawn_origin(Origin::NeverRespond));
        let cancellation = CancellationToken::new();
        cancellation.cancel();
        let outcome = runtime.block_on(transport(addr, Duration::from_secs(5), 1024).send_context(
            leg(),
            cancellation,
            Duration::from_secs(5),
        ));
        assert!(matches!(outcome, Err(LegFailure::Unavailable { .. })));
        server.abort();
    }

    /// The transport is usable behind the trait object the dispatcher holds.
    #[test]
    fn the_transport_is_usable_as_a_shared_trait_object() {
        let (addr, server) = runtime().block_on(spawn_origin(Origin::Status(500)));
        let boxed: Arc<dyn LegTransport> = Arc::new(transport(addr, Duration::from_secs(5), 1024));
        assert_eq!(Arc::strong_count(&boxed), 1);
        server.abort();
    }
}
